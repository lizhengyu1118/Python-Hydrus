# -*- coding: utf-8 -*-
"""
Custom module to parse HYDRUS 2D/3D ASCII output files.
This parser is specifically designed for the 'time-blocked' file format
(like H.TXT, V.TXT) and the MESHTRIA.TXT mesh format.

All data is loaded into RAM for fast interactive access.
All code and comments are in English as requested.
"""

import os
import numpy as np
import pyvista as pv
from datetime import datetime, timedelta

class HydrusModel:
    """
    Manages loading, parsing, and storing HYDRUS model data.
    Loads all data from a specified simulation folder into memory.
    """
    
    def __init__(self, simulation_folder_path):
        """
        Initializes the model loader with the path to a simulation folder.
        
        Args:
            simulation_folder_path (str): The path to the folder
                                          (e.g., "./Scenario_A").
        """
        self.folder_path = simulation_folder_path
        self.mesh_file = "MESHTRIA.TXT"
        self.head_file = "H.TXT"
        self.theta_file = "TH.TXT"
        self.velocity_file = "V.TXT"
        
        self.start_date = datetime(2004, 7, 8)
        self.node_count = 0
        self.element_count = 0
        
        self.mesh = None
        self.times = None  # Original time values (e.g., 0, 1, 7, ...)
        self.dates = None  # Datetime objects for the GUI
        
        # This dictionary will hold all time-series data in RAM
        # Keys: 'H', 'TH', 'Vx', 'Vy', 'Vz', 'V_mag'
        # Values: np.ndarray of shape (N_times, N_nodes)
        self.data_in_memory = {}
        
        # Store the last read line for user verification
        self._last_read_node_line = ""
        
        print(f"HydrusModel initialized for: {self.folder_path}")

    def load_all_data(self):
        """
        Public method to orchestrate the entire loading process.
        
        MODIFICATION: This function now returns True on success and False on failure,
        and includes granular error checking for each step.
        """
        print("--- Starting Data Load Process ---")
        
        # --- 1. Load mesh and perform user verification step ---
        try:
            print(f"Parsing {self.mesh_file}...")
            self.mesh = self._read_mesh()
            
            print("--- Verification Step ---")
            print(f"Successfully read {self.node_count} nodes.")
            
            # MODIFICATION: Removed interactive input() prompt
            # The script will now proceed automatically.
            print("Proceeding to load time-series data...")
            
        except FileNotFoundError as e:
            print(f"[ERROR] Critical file not found: {e.filename}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during MESH parsing: {e}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False

        # --- 2. Load H.TXT ---
        try:
            print(f"Parsing {self.head_file}...")
            h_times, h_data = self._parse_simple_file(self.head_file)
            self.times = h_times # Assume all files share time steps from H
            self.data_in_memory['H'] = h_data
            if h_data is None:
                 raise ValueError(f"{self.head_file} parsing returned no data.")
        except Exception as e:
            print(f"[ERROR] Failed to parse {self.head_file}: {e}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False

        # --- 3. Load TH.TXT ---
        try:
            print(f"Parsing {self.theta_file}...")
            th_times, th_data = self._parse_simple_file(self.theta_file)
            self.data_in_memory['TH'] = th_data
            if th_data is None:
                 raise ValueError(f"{self.theta_file} parsing returned no data.")
        except Exception as e:
            print(f"[ERROR] Failed to parse {self.theta_file}: {e}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False

        # --- 4. Load V.TXT ---
        try:
            print(f"Parsing {self.velocity_file}...")
            v_times, v_data_dict = self._parse_velocity_file(self.velocity_file)
            if not v_data_dict or v_data_dict.get('Vx') is None:
                raise ValueError(f"{self.velocity_file} parsing returned no data.")
            self.data_in_memory['Vx'] = v_data_dict['Vx']
            self.data_in_memory['Vy'] = v_data_dict['Vy']
            self.data_in_memory['Vz'] = v_data_dict['Vz']
        except Exception as e:
            print(f"[ERROR] Failed to parse {self.velocity_file}: {e}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False
            
        # --- 5. Calculate Velocity Magnitude ---
        try:
            print("Calculating velocity magnitude...")
            self._calculate_velocity_magnitude()
        except Exception as e:
            print(f"[ERROR] Failed during velocity magnitude calculation: {e}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False

        # --- 6. Generate Dates ---
        try:
            print("Generating date objects...")
            self._generate_dates()
        except Exception as e:
            print(f"[ERROR] Failed during date generation: {e}")
            print("Aborting data load.")
            return False # <-- MODIFICATION: Added return False

        # --- 7. Success ---
        print("--- Data Load Process Complete ---")
        return True # <-- MODIFICATION: This is the main fix

    def _find_header_index(self, lines, start_search_index, keywords):
        """
        Helper function to find a header line containing all keywords.
        
        Args:
            lines (list): All lines from the file.
            start_search_index (int): Index from where to start searching.
            keywords (list): List of strings (e.g., ['n', 'x', 'y', 'z'])
                             that must be in the header line.
                             
        Returns:
            int: The index of the header line.
            
        Raises:
            ValueError: If the header line cannot be found.
        """
        keywords_lower = [k.lower() for k in keywords]
        
        for i, line in enumerate(lines[start_search_index:], start=start_search_index):
            line_lower = line.lower()
            # Check if *all* keywords are present in this line
            if all(keyword in line_lower for keyword in keywords_lower):
                print(f"Found anchor line at index {i}: {line.strip()}")
                return i
        
        raise ValueError(f"Could not find anchor line containing keywords: {keywords}")

    def _read_mesh(self):
        """
        Parses the MESHTRIA.TXT file using the "Separator -> Header" logic.
        (Based on 10-24 9:11 PM and 10-24 9:10 PM specifications).
        """
        mesh_path = os.path.join(self.folder_path, self.mesh_file)
        
        with open(mesh_path, 'r') as f:
            lines = f.readlines()
        
        # --- 1. Find and Parse Node Block ---
        
        # MODIFICATION: Find anchors based on 10-24 9:11 PM logic
        # 1. Find Separator 1
        i_block_h_start = self._find_header_index(lines, 0, ["BLOCK H", "NODAL INFORMATION"])
        # 2. Find Header 1 (after Separator 1)
        i_node_header = self._find_header_index(lines, i_block_h_start + 1, ["n", "x", "y", "z"])
        # 3. Find Separator 2 (after Header 1)
        i_block_i_start = self._find_header_index(lines, i_node_header + 1, ["BLOCK I", "ELEMENT INFORMATION"])
        
        # 4. Extract data lines (between Header 1 and Separator 2)
        node_lines = lines[i_node_header + 1 : i_block_i_start]
        
        nodes_list = []
        self._last_read_node_line = "" # Clear last read line
        
        for line in node_lines:
            self._last_read_node_line = line.strip() # Store for verification
            if not self._last_read_node_line:
                continue # Skip empty lines
                
            try:
                # MODIFICATION: Robust parsing (10-24 8:44 PM spec)
                safe_line = line.replace(',', ' ')
                parts = safe_line.split()
                
                # Format: NodeID, X, Y, Z, ... (indices 1, 2, 3)
                if len(parts) >= 4:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    nodes_list.append([x, y, z])
                else:
                    pass # Skip sub-headers or malformed lines silently
                    
            except (ValueError, IndexError):
                print(f"Warning: Skipping non-numeric node line in {self.mesh_file}: {line.strip()}")
        
        nodes = np.array(nodes_list)
        
        # MODIFICATION: Dynamically set node count (10-24 9:02 PM spec)
        self.node_count = nodes.shape[0] 
        
        if self.node_count == 0:
             raise ValueError(f"Failed to parse any valid node coordinates between '{lines[i_node_header].strip()}' and '{lines[i_block_i_start].strip()}'.")

        # --- 2. Find and Parse Element Block ---
        
        # MODIFICATION: Find anchors based on 10-24 9:11 PM logic
        # 1. Separator 2 (i_block_i_start) is already known.
        # 2. Find Header 2 (after Separator 2)
        i_elem_header = self._find_header_index(lines, i_block_i_start + 1, 
                                                ["e", "i", "j", "k", "l", "m", "n", "o", "p", "sub"])
        i_elem_data_start = i_elem_header + 1
        
        # 3. Read to end of file (or next BLOCK)
        element_lines = []
        for line in lines[i_elem_data_start:]:
            if "*** BLOCK" in line:
                break
            element_lines.append(line)
        
        cells_list = []
        valid_element_count = 0
        max_node_index = self.node_count - 1 # 0-based index
        
        for line in element_lines:
            try:
                # MODIFICATION: Robust parsing
                safe_line = line.replace(',', ' ')
                parts_str = safe_line.split()
                parts = [int(p) for p in parts_str]
                
                # MODIFICATION: Force 4-node Tetra (10-24 9:18 PM spec)
                # Format: e, i, j, k, l, ... (min 5 columns)
                if len(parts) >= 5: 
                    # Get 4 nodes (indices 1 through 4)
                    node_indices = [p - 1 for p in parts[1:5]] # -1 for 0-based
                    
                    # Check if node indices are valid
                    if all(0 <= n <= max_node_index for n in node_indices):
                        # Format for VTK: [num_points, p0, p1, p2, p3]
                        cells_list.extend([4] + node_indices)
                        valid_element_count += 1
                    else:
                        print(f"Warning: Skipping element with out-of-bounds node index: {line.strip()}")
                else:
                    pass # Skip sub-headers or malformed lines silently
            except (ValueError, IndexError):
                 print(f"Warning: Skipping non-numeric element line in {self.mesh_file}: {line.strip()}")

        # --- 3. Construct Mesh ---
            
        # MODIFICATION: Fix 10-24 9:14 PM crash
        cells = np.array(cells_list, dtype=int)
        self.element_count = valid_element_count # Dynamically set element count
        
        if self.element_count == 0:
             print(f"Warning: Failed to parse any valid elements after '{lines[i_elem_header].strip()}'.")
             # We can still proceed if we only want to visualize nodes
             
        # MODIFICATION: Change cell type to TETRAHEDRON
        cell_types = np.full(self.element_count, pv.CellType.TETRA, dtype=np.uint8)
        
        print(f"Mesh Read: {self.node_count} nodes, {self.element_count} elements (parsed as 3D Tetrahedra).")
        
        # Create and return the PyVista mesh object
        return pv.UnstructuredGrid(cells, cell_types, nodes)

    def _parse_simple_file(self, filename):
        """
        Parses 'time-blocked' files like H.TXT and TH.TXT.
        (These files use space-separation and 'Time =' headers)
        """
        filepath = os.path.join(self.folder_path, filename)
        times = []
        data_blocks = []
        
        current_data = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Skip empty lines
                
                if "Time =" in line:
                    # 1. Finalize the previous data block
                    if current_data:
                        data_blocks.append(np.array(current_data))
                    
                    # 2. Start a new block
                    current_data = []
                    time_val = float(line.split('=')[-1].strip())
                    times.append(time_val)
                else:
                    # This is potentially a data line, add its values
                    try:
                        values = line.split()
                        float(values[0]) # Check first value
                        current_data.extend(list(map(float, values)))
                    except (ValueError, IndexError):
                        print(f"Warning: Skipping non-numeric line in {filename}: {line.strip()}")
        
        # Add the last block
        if current_data:
            data_blocks.append(np.array(current_data))
            
        if not data_blocks:
             # MODIFICATION: Return None instead of raising error immediately
             # This allows load_all_data to catch it.
             print(f"Error: No data blocks found in {filename}.")
             return np.array(times), None
             
        expected_nodes = self.node_count
        if expected_nodes == 0:
             raise ValueError("Node count is 0. Mesh parsing likely failed.")
             
        for i, block in enumerate(data_blocks):
            if block.shape[0] != expected_nodes:
                print(f"[Warning] Time index {i} in {filename} has {block.shape[0]} values, expected {expected_nodes}.")
                valid_block = np.zeros(expected_nodes)
                block_1d = block.flatten()
                len_to_copy = min(len(block_1d), expected_nodes)
                valid_block[:len_to_copy] = block_1d[:len_to_copy]
                data_blocks[i] = valid_block

        return np.array(times), np.vstack(data_blocks)

    def _parse_velocity_file(self, filename):
        """
        Parses the complex 'time-blocked' V.TXT file (Vx, Vy, Vz).
        (This file also uses space-separation and 'Time =' headers)
        """
        filepath = os.path.join(self.folder_path, filename)
        times = []
        vx_blocks, vy_blocks, vz_blocks = [], [], []
        
        current_data = []
        current_component = 0 # 0=None, 1=Vx, 2=Vy, 3=Vz
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if "Time =" in line:
                    if current_data:
                        block = np.array(current_data)
                        if current_component == 1:
                            vx_blocks.append(block)
                        elif current_component == 2:
                            vy_blocks.append(block)
                        elif current_component == 3:
                            vz_blocks.append(block)
                    
                    current_data = []
                    time_val = float(line.split('=')[-1].split()[0].strip())
                    
                    if "first component" in line:
                        current_component = 1
                        times.append(time_val)
                    elif "second component" in line:
                        current_component = 2
                    elif "third component" in line:
                        current_component = 3
                
                else:
                    try:
                        values = line.split()
                        float(values[0]) # Check first value
                        current_data.extend(list(map(float, values)))
                    except (ValueError, IndexError):
                        print(f"Warning: Skipping non-numeric line in {filename}: {line.strip()}")
        
        if current_data and current_component == 3:
            vz_blocks.append(np.array(current_data))
        
        if not (len(vx_blocks) == len(vy_blocks) == len(vz_blocks)):
            # MODIFICATION: Return empty dict instead of raising error
            print("Error: Velocity components have mismatched time steps.")
            return np.array(times), {}
        
        expected_nodes = self.node_count
        if expected_nodes == 0:
             raise ValueError("Node count is 0. Mesh parsing likely failed.")
             
        def verify_v_blocks(blocks, name):
            verified_blocks = []
            for i, block in enumerate(blocks):
                if block.shape[0] != expected_nodes:
                    print(f"[Warning] Time index {i} for {name} in {filename} has {block.shape[0]} values, expected {expected_nodes}.")
                    valid_block = np.zeros(expected_nodes)
                    block_1d = block.flatten()
                    len_to_copy = min(len(block_1d), expected_nodes)
                    valid_block[:len_to_copy] = block_1d[:len_to_copy]
                    verified_blocks.append(valid_block)
                else:
                    verified_blocks.append(block)
            
            # MODIFICATION: Handle empty blocks case
            if not verified_blocks:
                return None
            return np.vstack(verified_blocks)
        
        verified_vx = verify_v_blocks(vx_blocks, "Vx")
        verified_vy = verify_v_blocks(vy_blocks, "Vy")
        verified_vz = verify_v_blocks(vz_blocks, "Vz")

        if verified_vx is None or verified_vy is None or verified_vz is None:
            print(f"Error: Failed to verify velocity blocks (likely empty) in {filename}")
            return np.array(times), {}

        return np.array(times), {
            "Vx": verified_vx,
            "Vy": verified_vy,
            "Vz": verified_vz
        }

    def _calculate_velocity_magnitude(self):
        """
        Calculates the velocity magnitude from Vx, Vy, Vz
        and stores it in the data_in_memory dictionary.
        """
        if not all(k in self.data_in_memory for k in ('Vx', 'Vy', 'Vz')):
            print("Warning: Missing one or more velocity components. Skipping magnitude calculation.")
            # MODIFICATION: Ensure times exists before trying to get len
            num_times = len(self.times) if self.times is not None else 0
            self.data_in_memory['V_mag'] = np.zeros((num_times, self.node_count))
            return
            
        vx = self.data_in_memory['Vx']
        vy = self.data_in_memory['Vy']
        vz = self.data_in_memory['Vz']
        
        magnitude = np.sqrt(np.square(vx) + np.square(vy) + np.square(vz))
        self.data_in_memory['V_mag'] = magnitude

    def _generate_dates(self):
        """
        Creates a list of datetime objects from the time steps (days)
        and the start date.
        """
        if self.times is None:
            # MODIFICATION: Raise error to be caught by load_all_data
            raise ValueError("No time data loaded (self.times is None). Cannot generate dates.")
            
        self.dates = [self.start_date + timedelta(days=t) for t in self.times]
    
    # --- Public Getter Methods ---
    
    def get_mesh(self):
        return self.mesh
        
    def get_dates(self):
        return self.dates
        
    def get_data_by_name(self, data_name="H"):
        """
        Returns the full (N_times, N_nodes) data array from memory.
        """
        return self.data_in_memory.get(data_name)

