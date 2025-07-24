import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
from reducto import Reducto, AsyncReducto
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class ProcessingStrategy:
    ocr_mode: str
    array_extract: bool
    model_preference: str
    generate_citations: bool
    use_async: bool
    estimated_cost_multiplier: float

@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    confidence_scores: Dict[str, Any]
    processing_time: float
    strategy_used: ProcessingStrategy
    citations: Optional[List[Dict]] = None
    low_confidence_fields: Optional[List[str]] = None

class StructuredExtractionSystem:
    """
    Production-ready system for converting unstructured text to structured JSON
    following desired schemas with minimal constraints.
    
    Handles:
    - P1: Complex schemas (3-7 nesting levels, 50-150 objects, 1000+ literals)
    - P2: Large documents (50+ pages, up to 10MB)
    - P3: Adaptive processing based on complexity
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = Reducto(api_key=os.getenv("REDUCTO_API_KEY"))
        self.async_client = AsyncReducto(api_key=os.getenv("REDUCTO_API_KEY"))
        
        # Processing thresholds
        self.COMPLEXITY_THRESHOLDS = {
            'nesting_levels': {'medium': 3, 'high': 5},
            'total_fields': {'medium': 50, 'high': 100},
            'enum_count': {'medium': 100, 'high': 500},
            'document_pages': {'medium': 20, 'high': 50}
        }
        
    def extract_structured_data(
        self, 
        document_url: str, 
        schema: Dict[str, Any],
        system_prompt: str = "Be precise and thorough. Extract all relevant information from the document.",
        force_strategy: Optional[ProcessingStrategy] = None
    ) -> ExtractionResult:
        """
        Main extraction method that handles the complete pipeline.
        
        Args:
            document_url: URL to document (can be public URL, presigned S3 URL, or reducto:// URL)
            schema: JSON schema defining the desired output structure
            system_prompt: Instructions for the extraction model
            force_strategy: Override automatic strategy selection
            
        Returns:
            ExtractionResult with extracted data and metadata
        """
        start_time = time.time()
        
        # Analyze document and schema complexity
        doc_complexity = self._analyze_document_complexity(document_url)
        schema_complexity = self._analyze_schema_complexity(schema)
        
        # Determine processing strategy
        strategy = force_strategy or self._determine_processing_strategy(
            schema_complexity, doc_complexity
        )
        
        logger.info(f"Processing with strategy: {strategy}")
        
        # Execute extraction based on strategy
        if strategy.use_async:
            result = asyncio.run(self._extract_async(document_url, schema, system_prompt, strategy))
        else:
            result = self._extract_sync(document_url, schema, system_prompt, strategy)
        
        processing_time = time.time() - start_time
        
        # Process results and identify low confidence fields
        low_confidence_fields = self._identify_low_confidence_fields(result)
        
        return ExtractionResult(
            data=result.get('result', {}),
            confidence_scores=self._extract_confidence_scores(result),
            processing_time=processing_time,
            strategy_used=strategy,
            citations=result.get('citations', []),
            low_confidence_fields=low_confidence_fields
        )
    
    def _analyze_schema_complexity(self, schema: Dict[str, Any]) -> ProcessingComplexity:
        """Analyze JSON schema complexity to determine processing approach."""
        
        def count_nesting_levels(obj: Dict, level: int = 0) -> int:
            max_level = level
            if isinstance(obj, dict):
                if 'properties' in obj:
                    for prop in obj['properties'].values():
                        max_level = max(max_level, count_nesting_levels(prop, level + 1))
                elif 'items' in obj and isinstance(obj['items'], dict):
                    max_level = max(max_level, count_nesting_levels(obj['items'], level + 1))
            return max_level
        
        def count_total_fields(obj: Dict) -> int:
            count = 0
            if isinstance(obj, dict):
                if 'properties' in obj:
                    count += len(obj['properties'])
                    for prop in obj['properties'].values():
                        count += count_total_fields(prop)
                elif 'items' in obj and isinstance(obj['items'], dict):
                    count += count_total_fields(obj['items'])
            return count
        
        def count_enums(obj: Dict) -> int:
            count = 0
            if isinstance(obj, dict):
                if 'enum' in obj:
                    count += len(obj['enum'])
                if 'properties' in obj:
                    for prop in obj['properties'].values():
                        count += count_enums(prop)
                elif 'items' in obj and isinstance(obj['items'], dict):
                    count += count_enums(obj['items'])
            return count
        
        nesting_levels = count_nesting_levels(schema)
        total_fields = count_total_fields(schema)
        enum_count = count_enums(schema)
        
        logger.info(f"Schema analysis: {nesting_levels} levels, {total_fields} fields, {enum_count} enums")
        
        # Determine complexity based on thresholds
        if (nesting_levels >= self.COMPLEXITY_THRESHOLDS['nesting_levels']['high'] or
            total_fields >= self.COMPLEXITY_THRESHOLDS['total_fields']['high'] or
            enum_count >= self.COMPLEXITY_THRESHOLDS['enum_count']['high']):
            return ProcessingComplexity.HIGH
        elif (nesting_levels >= self.COMPLEXITY_THRESHOLDS['nesting_levels']['medium'] or
              total_fields >= self.COMPLEXITY_THRESHOLDS['total_fields']['medium'] or
              enum_count >= self.COMPLEXITY_THRESHOLDS['enum_count']['medium']):
            return ProcessingComplexity.MEDIUM
        else:
            return ProcessingComplexity.LOW
    
    def _analyze_document_complexity(self, document_url: str) -> ProcessingComplexity:
        """Analyze document to estimate processing complexity."""
        try:
            # First, parse the document to understand its structure
            parse_result = self.client.parse.run(
                document_url=document_url,
                options={"chunking": {"chunk_mode": "variable"}}
            )
            
            num_pages = parse_result.usage.num_pages
            
            logger.info(f"Document analysis: {num_pages} pages")
            
            if num_pages >= self.COMPLEXITY_THRESHOLDS['document_pages']['high']:
                return ProcessingComplexity.HIGH
            elif num_pages >= self.COMPLEXITY_THRESHOLDS['document_pages']['medium']:
                return ProcessingComplexity.MEDIUM
            else:
                return ProcessingComplexity.LOW
                
        except Exception as e:
            logger.warning(f"Could not analyze document complexity: {e}")
            return ProcessingComplexity.MEDIUM  # Default to medium if analysis fails
    
    def _determine_processing_strategy(
        self, 
        schema_complexity: ProcessingComplexity, 
        doc_complexity: ProcessingComplexity
    ) -> ProcessingStrategy:
        """Determine optimal processing strategy based on complexity analysis."""
        
        # Combine complexities (take the higher one)
        overall_complexity = max(schema_complexity, doc_complexity, key=lambda x: x.value)
        
        if overall_complexity == ProcessingComplexity.HIGH:
            return ProcessingStrategy(
                ocr_mode="agentic",  # Higher accuracy for complex cases
                array_extract=True,  # Handle long documents
                model_preference="accurate",
                generate_citations=True,  # For verification
                use_async=True,  # Prevent timeouts
                estimated_cost_multiplier=2.5
            )
        elif overall_complexity == ProcessingComplexity.MEDIUM:
            return ProcessingStrategy(
                ocr_mode="standard",
                array_extract=True,
                model_preference="balanced",
                generate_citations=True,
                use_async=False,
                estimated_cost_multiplier=1.5
            )
        else:  # LOW complexity
            return ProcessingStrategy(
                ocr_mode="standard",
                array_extract=False,
                model_preference="fast",
                generate_citations=False,
                use_async=False,
                estimated_cost_multiplier=1.0
            )
    
    def _extract_sync(
        self, 
        document_url: str, 
        schema: Dict[str, Any], 
        system_prompt: str, 
        strategy: ProcessingStrategy
    ) -> Dict[str, Any]:
        """Synchronous extraction for simpler cases."""
        
        config = {
            "document_url": document_url,
            "schema": schema,
            "system_prompt": system_prompt,
            "options": {
                "ocr_mode": strategy.ocr_mode
            },
            "generate_citations": strategy.generate_citations
        }
        
        if strategy.array_extract:
            config["array_extract"] = {"enabled": True}
        
        return self.client.extract.run(**config).model_dump()
    
    async def _extract_async(
        self, 
        document_url: str, 
        schema: Dict[str, Any], 
        system_prompt: str, 
        strategy: ProcessingStrategy
    ) -> Dict[str, Any]:
        """Asynchronous extraction for complex cases."""
        
        config = {
            "document_url": document_url,
            "schema": schema,
            "system_prompt": system_prompt,
            "options": {
                "ocr_mode": strategy.ocr_mode
            },
            "generate_citations": strategy.generate_citations
        }
        
        if strategy.array_extract:
            config["array_extract"] = {"enabled": True}
        
        result = await self.async_client.extract.run(**config)
        return result.model_dump()
    
    def _identify_low_confidence_fields(self, result: Dict[str, Any]) -> List[str]:
        """Identify fields that may need human review based on confidence scores."""
        low_confidence_fields = []
        
        def check_confidence(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if this field has low confidence
                    # This is a heuristic - you might need to adjust based on actual confidence data
                    if isinstance(value, str) and (
                        value.lower() in ['unknown', 'unclear', 'not found', ''] or
                        len(value.strip()) == 0
                    ):
                        low_confidence_fields.append(current_path)
                    
                    check_confidence(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_confidence(item, f"{path}[{i}]")
        
        if 'result' in result:
            check_confidence(result['result'])
        
        return low_confidence_fields
    
    def _extract_confidence_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confidence scores from the result."""
        # This would need to be implemented based on actual confidence data structure
        # For now, return empty dict as placeholder
        return {}
    
    def process_large_document_with_splitting(
        self, 
        document_url: str, 
        schema: Dict[str, Any],
        split_descriptions: List[Dict[str, str]],
        system_prompt: str = "Be precise and thorough."
    ) -> Dict[str, List[ExtractionResult]]:
        """
        Handle very large documents by splitting them first, then extracting from each section.
        
        Args:
            document_url: URL to the document
            schema: JSON schema for extraction
            split_descriptions: List of section descriptions for splitting
            system_prompt: Instructions for extraction
            
        Returns:
            Dictionary mapping section names to extraction results
        """
        # First, parse the document
        parse_result = self.client.parse.run(
            document_url=document_url,
            options={"chunking": {"chunk_mode": "variable"}}
        )
        
        # Split the document into sections
        split_result = self.client.split.run(
            document_url=f"jobid://{parse_result.job_id}",
            split_description=split_descriptions
        )
        
        results = {}
        
        # Extract from each section
        for split in split_result.result.splits:
            section_name = split.name
            pages = split.pages
            
            if pages:  # Only process if there are pages
                # Extract from this section with page range
                section_result = self.extract_structured_data(
                    document_url=f"jobid://{parse_result.job_id}",
                    schema=schema,
                    system_prompt=system_prompt
                )
                
                results[section_name] = section_result
        
        return results
    
    def upload_and_extract(
        self, 
        file_path: Union[str, Path], 
        schema: Dict[str, Any],
        system_prompt: str = "Be precise and thorough."
    ) -> ExtractionResult:
        """
        Upload a local file and extract structured data from it.
        
        Args:
            file_path: Path to the local file
            schema: JSON schema for extraction
            system_prompt: Instructions for extraction
            
        Returns:
            ExtractionResult with extracted data
        """
        # Upload the file
        upload_result = self.client.upload(file=Path(file_path))
        
        # Extract structured data
        return self.extract_structured_data(
            document_url=upload_result,
            schema=schema,
            system_prompt=system_prompt
        )

# Example usage and test cases
def create_complex_schema() -> Dict[str, Any]:
    """Create a complex schema for testing (150k tokens worth of complexity)."""
    return {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "additionalProperties": False,
  "definitions": {
    "iso8601": {
      "type": "string",
      "description": "e.g. 2014-06-29",
      "pattern": "^([1-2][0-9]{3}-[0-1][0-9]-[0-3][0-9]|[1-2][0-9]{3}-[0-1][0-9]|[1-2][0-9]{3})$"
    }
  },
  "properties": {
    "$schema": {
      "type": "string",
      "description": "link to the version of the schema that can validate the resume",
      "format": "uri"
    },
    "basics": {
      "type": "object",
      "additionalProperties": True,
      "properties": {
        "name": {
          "type": "string"
        },
        "label": {
          "type": "string",
          "description": "e.g. Web Developer"
        },
        "image": {
          "type": "string",
          "description": "URL (as per RFC 3986) to a image in JPEG or PNG format"
        },
        "email": {
          "type": "string",
          "description": "e.g. thomas@gmail.com",
          "format": "email"
        },
        "phone": {
          "type": "string",
          "description": "Phone numbers are stored as strings so use any format you like, e.g. 712-117-2923"
        },
        "url": {
          "type": "string",
          "description": "URL (as per RFC 3986) to your website, e.g. personal homepage",
          "format": "uri"
        },
        "summary": {
          "type": "string",
          "description": "Write a short 2-3 sentence biography about yourself"
        },
        "location": {
          "type": "object",
          "additionalProperties": True,
          "properties": {
            "address": {
              "type": "string",
              "description": "To add multiple address lines, use \n. For example, 1234 Glücklichkeit Straße\nHinterhaus 5. Etage li."
            },
            "postalCode": {
              "type": "string"
            },
            "city": {
              "type": "string"
            },
            "countryCode": {
              "type": "string",
              "description": "code as per ISO-3166-1 ALPHA-2, e.g. US, AU, IN"
            },
            "region": {
              "type": "string",
              "description": "The general region where you live. Can be a US state, or a province, for instance."
            }
          }
        },
        "profiles": {
          "type": "array",
          "description": "Specify any number of social networks that you participate in",
          "additionalItems": False,
          "items": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
              "network": {
                "type": "string",
                "description": "e.g. Facebook or Twitter"
              },
              "username": {
                "type": "string",
                "description": "e.g. neutralthoughts"
              },
              "url": {
                "type": "string",
                "description": "e.g. http://twitter.example.com/neutralthoughts",
                "format": "uri"
              }
            }
          }
        }
      }
    },
    "work": {
      "type": "array",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. Facebook"
          },
          "location": {
            "type": "string",
            "description": "e.g. Menlo Park, CA"
          },
          "description": {
            "type": "string",
            "description": "e.g. Social Media Company"
          },
          "position": {
            "type": "string",
            "description": "e.g. Software Engineer"
          },
          "url": {
            "type": "string",
            "description": "e.g. http://facebook.example.com",
            "format": "uri"
          },
          "startDate": {
            "$ref": "#/definitions/iso8601"
          },
          "endDate": {
            "$ref": "#/definitions/iso8601"
          },
          "summary": {
            "type": "string",
            "description": "Give an overview of your responsibilities at the company"
          },
          "highlights": {
            "type": "array",
            "description": "Specify multiple accomplishments",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. Increased profits by 20% from 2011-2012 through viral advertising"
            }
          }
        }
      }
    },
    "volunteer": {
      "type": "array",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "organization": {
            "type": "string",
            "description": "e.g. Facebook"
          },
          "position": {
            "type": "string",
            "description": "e.g. Software Engineer"
          },
          "url": {
            "type": "string",
            "description": "e.g. http://facebook.example.com",
            "format": "uri"
          },
          "startDate": {
            "$ref": "#/definitions/iso8601"
          },
          "endDate": {
            "$ref": "#/definitions/iso8601"
          },
          "summary": {
            "type": "string",
            "description": "Give an overview of your responsibilities at the company"
          },
          "highlights": {
            "type": "array",
            "description": "Specify accomplishments and achievements",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. Increased profits by 20% from 2011-2012 through viral advertising"
            }
          }
        }
      }
    },
    "education": {
      "type": "array",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "institution": {
            "type": "string",
            "description": "e.g. Massachusetts Institute of Technology"
          },
          "url": {
            "type": "string",
            "description": "e.g. http://facebook.example.com",
            "format": "uri"
          },
          "area": {
            "type": "string",
            "description": "e.g. Arts"
          },
          "studyType": {
            "type": "string",
            "description": "e.g. Bachelor"
          },
          "startDate": {
            "$ref": "#/definitions/iso8601"
          },
          "endDate": {
            "$ref": "#/definitions/iso8601"
          },
          "score": {
            "type": "string",
            "description": "grade point average, e.g. 3.67/4.0"
          },
          "courses": {
            "type": "array",
            "description": "List notable courses/subjects",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. H1302 - Introduction to American history"
            }
          }
        }
      }
    },
    "awards": {
      "type": "array",
      "description": "Specify any awards you have received throughout your professional career",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "title": {
            "type": "string",
            "description": "e.g. One of the 100 greatest minds of the century"
          },
          "date": {
            "$ref": "#/definitions/iso8601"
          },
          "awarder": {
            "type": "string",
            "description": "e.g. Time Magazine"
          },
          "summary": {
            "type": "string",
            "description": "e.g. Received for my work with Quantum Physics"
          }
        }
      }
    },
    "certificates": {
      "type": "array",
      "description": "Specify any certificates you have received throughout your professional career",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. Certified Kubernetes Administrator"
          },
          "date": {
            "type": "string",
            "description": "e.g. 1989-06-12",
            "format": "date"
          },
          "url": {
            "type": "string",
            "description": "e.g. http://example.com",
            "format": "uri"
          },
          "issuer": {
            "type": "string",
            "description": "e.g. CNCF"
          }
        }
      }
    },
    "publications": {
      "type": "array",
      "description": "Specify your publications through your career",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. The World Wide Web"
          },
          "publisher": {
            "type": "string",
            "description": "e.g. IEEE, Computer Magazine"
          },
          "releaseDate": {
            "$ref": "#/definitions/iso8601"
          },
          "url": {
            "type": "string",
            "description": "e.g. http://www.computer.org.example.com/csdl/mags/co/1996/10/rx069-abs.html",
            "format": "uri"
          },
          "summary": {
            "type": "string",
            "description": "Short summary of publication. e.g. Discussion of the World Wide Web, HTTP, HTML."
          }
        }
      }
    },
    "skills": {
      "type": "array",
      "description": "List out your professional skill-set",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. Web Development"
          },
          "level": {
            "type": "string",
            "description": "e.g. Master"
          },
          "keywords": {
            "type": "array",
            "description": "List some keywords pertaining to this skill",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. HTML"
            }
          }
        }
      }
    },
    "languages": {
      "type": "array",
      "description": "List any other languages you speak",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "language": {
            "type": "string",
            "description": "e.g. English, Spanish"
          },
          "fluency": {
            "type": "string",
            "description": "e.g. Fluent, Beginner"
          }
        }
      }
    },
    "interests": {
      "type": "array",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. Philosophy"
          },
          "keywords": {
            "type": "array",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. Friedrich Nietzsche"
            }
          }
        }
      }
    },
    "references": {
      "type": "array",
      "description": "List references you have received",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. Timothy Cook"
          },
          "reference": {
            "type": "string",
            "description": "e.g. Joe blogs was a great employee, who turned up to work at least once a week. He exceeded my expectations when it came to doing nothing."
          }
        }
      }
    },
    "projects": {
      "type": "array",
      "description": "Specify career projects",
      "additionalItems": False,
      "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "name": {
            "type": "string",
            "description": "e.g. The World Wide Web"
          },
          "description": {
            "type": "string",
            "description": "Short summary of project. e.g. Collated works of 2017."
          },
          "highlights": {
            "type": "array",
            "description": "Specify multiple features",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. Directs you close but not quite there"
            }
          },
          "keywords": {
            "type": "array",
            "description": "Specify special elements involved",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. AngularJS"
            }
          },
          "startDate": {
            "$ref": "#/definitions/iso8601"
          },
          "endDate": {
            "$ref": "#/definitions/iso8601"
          },
          "url": {
            "type": "string",
            "format": "uri",
            "description": "e.g. http://www.computer.org/csdl/mags/co/1996/10/rx069-abs.html"
          },
          "roles": {
            "type": "array",
            "description": "Specify your role on this project or in company",
            "additionalItems": False,
            "items": {
              "type": "string",
              "description": "e.g. Team Lead, Speaker, Writer"
            }
          },
          "entity": {
            "type": "string",
            "description": "Specify the relevant company/entity affiliations e.g. 'greenpeace', 'corporationXYZ'"
          },
          "type": {
            "type": "string",
            "description": " e.g. 'volunteering', 'presentation', 'talk', 'application', 'conference'"
          }
        }
      }
    },
    "meta": {
      "type": "object",
      "description": "The schema version and any other tooling configuration lives here",
      "additionalProperties": True,
      "properties": {
        "canonical": {
          "type": "string",
          "description": "URL (as per RFC 3986) to latest version of this document",
          "format": "uri"
        },
        "version": {
          "type": "string",
          "description": "A version field which follows semver - e.g. v1.0.0"
        },
        "lastModified": {
          "type": "string",
          "description": "Using ISO 8601 with YYYY-MM-DDThh:mm:ss"
        }
      }
    }
  },
  "title": "Resume Schema",
  "type": "object"
}

# def main():
#     """Example usage of the system."""
    
#     # Initialize the system
#     system = StructuredExtractionSystem()
    
#     # Create a complex schema (this would be your 150k token schema)
#     schema = create_complex_schema()
    
#     # Example 1: Extract from a URL
#     # try:
#     #     result = system.extract_structured_data(
#     #         document_url="https://example.com/complex-document.pdf",
#     #         schema=schema,
#     #         system_prompt="Extract all negotiation requirements and stakeholder information. Pay special attention to financial terms and status changes."
#     #     )
        
#     #     print(f"Extraction completed in {result.processing_time:.2f} seconds")
#     #     print(f"Strategy used: {result.strategy_used}")
#     #     print(f"Low confidence fields: {result.low_confidence_fields}")
#     #     print(f"Extracted data: {json.dumps(result.data, indent=2)}")
        
#     # except Exception as e:
#     #     print(f"Extraction failed: {e}")
    
#     # Example 2: Upload and extract from local file
#     try:
#         local_result = system.upload_and_extract(
#             file_path="/content/ShreyaParsewar_Resume.pdf",
#             schema=schema
#         )
#         print(f"Local file extraction completed: {local_result.processing_time:.2f}s")
#         print(f"Strategy used: {local_result.strategy_used}")
#         print(f"Low confidence fields: {local_result.low_confidence_fields}")
#         print(f"Extracted data: {json.dumps(local_result.data, indent=2)}")
#     except Exception as e:
#         print(f"Local extraction failed: {e}")

# if __name__ == "__main__":
#     main()