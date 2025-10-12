# C-5 Binary Matrix Dataset Description

## Overview
The `c5_Matrix.csv` dataset is a comprehensive quantum logic matrix containing binary representations of quantum values (QV) across multiple events. This dataset serves as the foundation for pattern recognition and prediction in the C-5 matrix project.

## Dataset Structure

### Format
- **File Type**: CSV (Comma Separated Values)
- **Total Events**: Approximately 11,581 entries
- **Format**: Wide-format binary representation
- **Encoding**: ASCII/UTF-8 compatible
- **Memory footprint**: ~4.0 MB

## Data Characteristics
1. **Event ID** (Column 1): Unique identifier for each event, appearing to be sequential numbers (e.g., 11448, 11449, etc.)
2. **Quantum Value Binary Matrix** (Columns 2-40): 39 binary columns representing the activation status of each Quantum Value (QV) position:
   - Each row contains exactly 5 active QVs (five '1' values) in the QV columns (2-40)
   - Value `1`: Indicates an active QV at this position
   - Value `0`: Indicates an inactive QV at this position
3. **Binary Matrix/Representation (Columns 2-40)**: One-hot encoded representation of active quantum values
4. **Cylindrical Adjacency Consideration**: The wrap-around (QV_39 adjacent to QV_1) implies a cyclic structure, like a ring lattice. This can influence mappings, e.g., using periodic boundary conditions in quantum models.

### Columns
event-ID,QV_1,QV_2,QV_3,QV_4,QV_5,QV_6,QV_7,QV_8,QV_9,QV_10,QV_11,QV_12,QV_13,QV_14,QV_15,QV_16,QV_17,QV_18,QV_19,QV_20,QV_21,QV_22,QV_23,QV_24,QV_25,QV_26,QV_27,QV_28,QV_29,QV_30,QV_31,QV_32,QV_33,QV_34,QV_35,QV_36,QV_37,QV_38,QV_39


 


