from typing import Any, Dict, Optional

def success_response(
    recommendations: Any,
    model_name: str,
    version: Optional[str] = None,
    message: str = "Recommendations retrieved successfully",
    **kwargs
) -> Dict[str, Any]:
    """
    Standardized API success response wrapper for recommendation endpoints.
    """
    if not isinstance(recommendations, list):
        recs_list = []
    else:
        recs_list = [r for r in recommendations if r is not None]

    payload = {
        "status": "success",
        "message": message,
        "data": {
            "recommendations": recs_list
        },
        "meta": {
            "model": model_name,
            "version": version or "1.0",
            "count": len(recs_list)
        }
    }
    
    # Add backward compatibility fields
    payload["recommendations"] = recs_list
    payload["results"] = recs_list
    payload["count"] = len(recs_list)

    # Blend with any other extra fields (like query, explain, weights, etc.)
    for k, v in kwargs.items():
        if k not in payload:
            payload[k] = v

    return payload

def error_response(
    message: str,
    model_name: str,
    version: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Standardized API error response wrapper for recommendation endpoints.
    """
    payload = {
        "status": "error",
        "message": message,
        "data": {
            "recommendations": []
        },
        "meta": {
            "model": model_name,
            "version": version or "1.0",
            "count": 0
        }
    }
    
    # Add backward compatibility fields
    payload["recommendations"] = []
    payload["results"] = []
    payload["count"] = 0

    # Blend with any other extra fields
    for k, v in kwargs.items():
        if k not in payload:
            payload[k] = v

    return payload
