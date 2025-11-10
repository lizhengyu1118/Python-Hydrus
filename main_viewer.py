# -*- coding: utf-8 -*-
"""
Main Interactive 3D Viewer for HYDRUS Results.

This program scans for simulation folders, prompts the user to select one,
prompts for the initial data type to display ('H', 'TH', 'V_mag', 'V_vector'),
and then launches a PyVista interactive window.

The window provides:
1. A slider to select the date (linked to simulation time steps).
2. The ability to view the data type selected at startup.

Relies on the 'hydrus_parser.py' module.
All code and comments are in English as requested.
"""
import notify_on_exit
import time
import os
import sys
import numpy as np
import pyvista as pv
from hydrus_parser import HydrusModel

class HydrusViewer:
    """
    Manages the PyVista Plotter and all GUI interactions.
    """

    def __init__(self, loader_instance, initial_data_key="H"):
        """
        Initialize the viewer with a pre-loaded HydrusModel object
        and the initial data type to display.
        """
        self.loader = loader_instance
        self.plotter = pv.Plotter(
            window_size=[1200, 800],
            notebook=False
        ) # Ensure it runs in a separate window

        self.mesh = self.loader.get_mesh()
        self.dates = self.loader.get_dates()
        
        # MODIFICATION: Add references for actors
        self.main_actor = None # Actor for scalar mesh
        self.glyph_actor = None # Actor for vector glyphs
        self.glyph_factor = 0.0 # Scale factor for arrows

        # State variables
        self.current_time_index = 0
        self.current_data_name = initial_data_key 

        # Data dictionary
        self.data_cache = {
            "H": self.loader.get_data_by_name("H"),
            "TH": self.loader.get_data_by_name("TH"),
            "V_mag": self.loader.get_data_by_name("V_mag")
        }
        
        # MODIFICATION: Combine Vx, Vy, Vz into a single vector entry
        if all(k in self.loader.data_in_memory for k in ('Vx', 'Vy', 'Vz')):
            print("Combining Vx, Vy, Vz into V_vector.")
            vx = self.loader.get_data_by_name("Vx")
            vy = self.loader.get_data_by_name("Vy")
            vz = self.loader.get_data_by_name("Vz")
            # Stack into (N_times, N_nodes, 3)
            self.data_cache['V_vector'] = np.stack((vx, vy, vz), axis=-1)
        
        # Check for data load issues
        if self.mesh is None:
            raise ValueError("Mesh was not loaded successfully.")
        if not self.dates:
            raise ValueError("Dates were not loaded successfully.")
        
        # Check if the requested data (scalar or vector) exists in the cache
        if self.current_data_name not in self.data_cache or self.data_cache[self.current_data_name] is None:
             raise ValueError(f"Selected data '{self.current_data_name}' was not loaded successfully.")

        # Store scalar bar range for consistent visualization
        self.scalar_bar_ranges = {}
        for key, data in self.data_cache.items():
            if data is not None and key != 'V_vector': # Scalar ranges only
                try:
                    min_val, max_val = np.nanmin(data), np.nanmax(data)
                    if np.isnan(min_val) or np.isnan(max_val):
                         self.scalar_bar_ranges[key] = [0, 1]
                    else:
                        self.scalar_bar_ranges[key] = [min_val, max_val]
                except Exception as e:
                    print(f"Warning: Error calculating range for {key}: {e}.")
                    self.scalar_bar_ranges[key] = [0, 1]

        # Internal key for plotting scalars
        self.internal_plot_key = 'data_to_plot'

    def _launch_scalar_view(self):
        """Helper function to set up the scalar data view."""
        print(f"Setting up scalar view for: {self.current_data_name}")
        # 1. Get initial data
        initial_data_full = self.data_cache[self.current_data_name]
        initial_data_slice = initial_data_full[0, :]

        # 2. Add the mesh actor
        self.main_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=initial_data_slice,
            name='main_mesh',
            scalar_bar_args={'title': self.current_data_name},
            clim=self.scalar_bar_ranges.get(self.current_data_name, [0, 1])
        )

        # 3. Rename arrays
        try:
            mesh_dataset = self.main_actor.mapper.dataset
            default_scalars_name = mesh_dataset.active_scalars_name
            if default_scalars_name and default_scalars_name != self.internal_plot_key:
                mesh_dataset.rename_array(default_scalars_name, self.internal_plot_key)
            mesh_dataset.set_active_scalars(self.internal_plot_key)
        except Exception as e:
            print(f"Warning: Failed to rename initial scalars: {e}")

    def _launch_vector_view(self):
        """Helper function to set up the vector (glyph) view."""
        print("Setting up vector view for: V_vector")
        
        # 1. Add the base mesh (static)
        self.main_actor = self.plotter.add_mesh(
            self.mesh, 
            style='wireframe', 
            color='grey', 
            opacity=0.3
        )
        
        # 2. Get initial vector data
        initial_vector_data = self.data_cache['V_vector'][0, :, :]
        
        # 3. Attach vectors to the (in-memory) mesh
        # We will update this mesh object directly in the update loop
        self.mesh.point_data['vectors'] = initial_vector_data
        self.mesh.set_active_vectors('vectors')
        
        # 4. Calculate a reasonable glyph factor
        # This might need tuning based on your model's scale
        self.glyph_factor = self.mesh.length * 0.01
        
        # 5. Generate and add the glyphs (arrows)
        glyphs = self.mesh.glyph(
            scale='vectors', 
            orient='vectors', 
            factor=self.glyph_factor,
            geom=pv.Arrow() # Use arrows
        )
        
        self.glyph_actor = self.plotter.add_mesh(
            glyphs, 
            scalar_bar_args={'title': 'Velocity Magnitude'}
        )

    def launch(self):
        """
        Initializes the scene and shows the interactive window.
        """
        if self.mesh.n_points == 0:
            print("Error: Mesh has 0 points. Cannot render.")
            return
        print(f"Setting up scene with {self.mesh.n_points} nodes and {self.mesh.n_cells} elements.")

        # MODIFICATION: Branch logic for scalar vs vector
        if self.current_data_name == 'V_vector':
            self._launch_vector_view()
        else:
            self._launch_scalar_view()
        
        # --- Add Common GUI Widgets ---

        self.date_label = self.plotter.add_text(
            f"Date: {self.dates[0].strftime('%Y-%m-%d')}",
            position='upper_left',
            font_size=12
        )

        self.plotter.add_slider_widget(
            callback=self._on_slider_update,
            rng=[0, len(self.dates) - 1],
            value=0,
            title="Date",
            fmt="Time Step %d",
            style='modern',
            pointa=(0.25, 0.9),
            pointb=(0.75, 0.9)
        )

        self.plotter.view_isometric()
        self.plotter.enable_zoom_style()
        self.plotter.enable_trackball_style() 

        print("Launching interactive viewer. Close the window to exit.")
        self.plotter.show()

    def _on_slider_update(self, value):
        """
        Callback for when the time slider is moved.
        """
        self.current_time_index = int(value)

        if hasattr(self, 'date_label'):
            self.date_label.SetText(2, f"Date: {self.dates[self.current_time_index].strftime('%Y-%m-%d')}")

        # Update the plot
        self._update_plot()

    # MODIFICATION: Split update logic
    def _update_scalar_plot(self):
        """Updates the scalars on the main mesh."""
        try:
            full_data = self.data_cache[self.current_data_name]
            new_slice = full_data[self.current_time_index, :]

            mesh_dataset = self.main_actor.mapper.dataset
            mesh_dataset.point_data[self.internal_plot_key] = new_slice
            
            self.main_actor.mapper.modified()
            self.plotter.render()
        except Exception as e:
            print(f"Error during scalar plot update: {e}")

    # MODIFICATION: New function to update glyphs
    def _update_vector_plot(self):
        """Updates the vector glyphs."""
        try:
            # 1. Get new vector data
            new_slice = self.data_cache['V_vector'][self.current_time_index, :, :]
            
            # 2. Update the base mesh's vector data
            self.mesh.point_data['vectors'] = new_slice
            
            # 3. Re-generate the glyphs from the updated base mesh
            new_glyphs = self.mesh.glyph(
                scale='vectors', 
                orient='vectors', 
                factor=self.glyph_factor,
                geom=pv.Arrow()
            )
            
            # 4. Overwrite the glyph actor's geometry
            self.glyph_actor.mapper.dataset.copy_from(new_glyphs)
            self.glyph_actor.mapper.modified()
            
            # 5. Re-render
            self.plotter.render()
        except Exception as e:
            print(f"Error during vector plot update: {e}")

    def _update_plot(self):
        """
        Central function to update the mesh's active scalars or vectors.
        """
        # MODIFICATION: Branch update logic
        if self.current_data_name == 'V_vector':
            self._update_vector_plot()
        else:
            self._update_scalar_plot()


# --- Main execution ---

def find_simulation_folders(root_dir):
    """
    Finds all subfolders in root_dir that contain a MESHTRIA.TXT file.
    """
    folders = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            mesh_file_found = False
            try:
                for file in os.listdir(entry.path):
                    if file.lower() == "meshtria.txt":
                        mesh_file_found = True
                        break
            except OSError:
                continue
            if mesh_file_found:
                 folders.append(entry.name)
    return sorted(folders)

def prompt_for_folder(folders):
    """
    Displays a menu and asks the user to select a folder.
    """
    print("\nFound the following simulation folders:")
    for i, folder in enumerate(folders, 1):
        print(f"  [{i}] {folder}")

    while True:
        try:
            choice = input(f"Which simulation do you want to load? (1-{len(folders)}): ")
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(folders):
                return folders[choice_index]
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# MODIFICATION: Update prompt to include synthetic 'V_vector' option
def prompt_for_data_type(loader):
    """
    Displays available data types and asks the user to select one.
    """
    # 1. Get standard scalar data
    available_data = sorted([
        k for k, v in loader.data_in_memory.items() 
        if v is not None and k in ['H', 'TH', 'V_mag']
    ])
    
    # 2. Check if vector components exist to create a 'V_vector' option
    has_vectors = all(
        k in loader.data_in_memory and loader.data_in_memory[k] is not None 
        for k in ('Vx', 'Vy', 'Vz')
    )
    
    if has_vectors:
        available_data.append("V_vector") # Add synthetic option

    if not available_data:
        print("Error: No data ('H', 'TH', 'V_mag', or 'V_vector') was successfully loaded.")
        return None

    print("\nAvailable data types to visualize:")
    for i, data_key in enumerate(available_data, 1):
        print(f"  [{i}] {data_key}")

    while True:
        try:
            choice = input(f"Which data type do you want to display initially? (1-{len(available_data)}): ")
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(available_data):
                return available_data[choice_index]
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# --- ADDED: Email Notification Import ---
try:
    import notify_on_exit
except ImportError:
    print("Warning: 'notify_on_exit.py' not found.")
    print("Email notification on completion will be disabled.")
    notify_on_exit = None # Define as None to handle gracefully
# --- End ADDED ---

SMTP_CONFIG = {
    "server": "smtp.gmail.com",       # Example for Gmail
    "port": 465,                      # Example for Gmail (SSL)
    "sender_email": "lizhengyu1118@gmail.com",
    "sender_password": "zcup lvjb ayre pxpd" 
}

RECIPIENT_EMAIL = "lizhengyu1118@gmail.com"

# --- End Configuration ---

def main():
    """
    Main execution script.
    """
 # --- ADDED: Register Email Notification ---
    if notify_on_exit: # Only register if module was imported successfully
        try:
            # Register the notification function *at the beginning*
            notify_on_exit.register_completion_notify(
                recipient_email=RECIPIENT_EMAIL,
                smtp_config=SMTP_CONFIG,
                program_name="Interactive Viewer (main_viewer.py)" # Specific name
            )
        except Exception as e:
            # Print warning to stderr
            print(f"Warning: Failed to register email notification: {e}", file=sys.stderr)
            # Continue execution even if email registration fails
    else:
        print("Skipping email notification registration (module not found).", file=sys.stderr)
    # --- End ADDED ---

    project_dir = os.path.dirname(os.path.abspath(__file__))

    sim_folders = find_simulation_folders(project_dir)
    if not sim_folders:
        print(f"Error: No simulation folders containing MESHTRIA.TXT found in {project_dir}")
        sys.exit(1)

    selected_folder_name = prompt_for_folder(sim_folders)
    if not selected_folder_name:
        print("No selection made. Exiting.")
        sys.exit(0)

    selected_folder_path = os.path.join(project_dir, selected_folder_name)

    print(f"Loading data from: {selected_folder_path}")
    loader = HydrusModel(selected_folder_path)

    if not loader.load_all_data():
        print("Data loading aborted by user or failed.")
        sys.exit(1)

    if loader.get_mesh() is None or not loader.get_dates():
        print("Data was not loaded correctly after verification. Exiting.")
        sys.exit(1)

    # MODIFICATION: This prompt now supports 'V_vector'
    selected_data_key = prompt_for_data_type(loader)
    if not selected_data_key:
        sys.exit(1) 

    print(f"Initializing viewer with data type: {selected_data_key}")
    viewer = HydrusViewer(loader, initial_data_key=selected_data_key)
    viewer.launch()

if __name__ == "__main__":
    main()

