"""
This module provides a dataclass for representing material properties in electromagnetic simulations.

And provides a method to parse material properties from a file. 
"""
import numpy as np
from dataclasses import dataclass, fields

@dataclass
class Material:
    """Class representing material properties for electromagnetic simulation."""
    fields_diffusively_scattered: float = 0.0
    cross_polarized_power: float = 0.0
    directive_alpha: int = 4.0
    directive_beta: int = 4.0
    directive_lambda: float = 0.5
    conductivity: float = 0.0
    permittivity: float = 1.0
    roughness: float = 0.0
    thickness: float = 0.0
    
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