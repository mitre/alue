def load_schema(schema_class_name: str):
    """Load Pydantic schema class dynamically."""
    if not schema_class_name:
        return None
        
    try:
        module_name, class_name = schema_class_name.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        schema = getattr(module, class_name)
        print(f"Using schema: {schema.__name__}")
        return schema
        
    except Exception as e:
        print(f"Warning: Could not import schema {schema_class_name}: {e}")
        return None



def parse_fields_to_extract(value):
    """Parse fields_to_extract argument - can be None, single string, or comma-separated list."""
    if value is None or value.lower() == 'none':
        return None
    elif ',' in value:
        return [field.strip() for field in value.split(',')]
    else:
        return value.strip()
    

def load_normalizer(normalizer_name: str):
    """Load normalization function dynamically."""
    if not normalizer_name:
        return None
        
    try:
        if '.' in normalizer_name:
            # Full module path provided
            module_name, func_name = normalizer_name.rsplit('.', 1)
        else:
            # Just function name, use default path
            module_name = "alue.output_normalizations"
            func_name = normalizer_name

        module = __import__(module_name, fromlist=[func_name])
        normalizer_func = getattr(module, func_name)
        print(f"Using normalizer: {normalizer_func.__name__}")
        return normalizer_func
        
    except Exception as e:
        print(f"Warning: Could not import normalizer {normalizer_name}: {e}")
        return None