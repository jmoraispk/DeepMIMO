import numpy as np
from dataclasses import dataclass, fields

@dataclass
class Material:
    """Class representing material properties for electromagnetic simulation."""
    fields_diffusively_scattered: float
    cross_polarized_power: float
    directive_alpha: int
    directive_beta: int
    directive_lambda: float
    conductivity: float
    permittivity: float
    roughness: float
    thickness: float
    
    @classmethod
    def from_file(cls, file_path):
        """Parse material properties from a file."""
        try:
            with open(file_path, "r") as f:
                file_lines = f.readlines()
        except FileNotFoundError:
            return None
        
        # Get field names and types from the Material dataclass
        material_attributes = {}
        for field in fields(cls):
            field_type = np.float64 if field.type is float else np.int64
            material_attributes[field.name] = field_type
        
        # Initialize a dictionary to store the parsed values
        parsed_values = {}
        
        # Parse the file
        for line in file_lines:
            for keyword, dtype in material_attributes.items():
                if line.startswith(keyword):
                    # Extract the value and convert to the appropriate type
                    value = dtype(line.split(" ")[-1].strip())
                    parsed_values[keyword] = value
                    break
        
        # Create the Material object by unpacking the dictionary
        return cls(**parsed_values) 