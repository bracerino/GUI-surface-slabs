import streamlit as st
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import surface, add_vacuum
from ase.constraints import FixAtoms
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import py3Dmol
import streamlit.components.v1 as components
from io import StringIO
import time

#from Databases_search import *


#show_database_search = st.checkbox("Enable database search",
#                                  value=False,
#                                   help="Enable to search in Materials Project, AFLOW, and COD databases")
#databases(show_database_search)

def pymatgen_to_ase(structure):
    symbols = [str(site.specie) for site in structure]
    positions = [site.coords for site in structure]
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def ase_to_pymatgen(atoms):
    from pymatgen.core import Lattice
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    lattice = Lattice(cell)
    return Structure(lattice, symbols, positions, coords_are_cartesian=True)


def get_orthogonal_cell(structure, max_atoms=200):
    from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        cell_params = ase_atoms.get_cell_lengths_and_angles()
        if len(cell_params) >= 6:
            angles = [cell_params[3], cell_params[4], cell_params[5]]
        else:
            angles = [90.0, 90.0, 90.0]

        angle_deviations = []
        for angle in angles:
            deviation = abs(float(angle) - 90.0)
            angle_deviations.append(deviation)

        is_already_orthogonal = True
        for deviation in angle_deviations:
            if deviation >= 1e-6:
                is_already_orthogonal = False
                break

        if is_already_orthogonal:
            return structure.copy()

        try:
            transformer = CubicSupercellTransformation(
                max_atoms=int(max_atoms),
                min_atoms=len(structure),
                force_90_degrees=True,
                allow_orthorhombic=True,
                angle_tolerance=0.1,
                min_length=5.0
            )
            orthogonal_structure = transformer.apply_transformation(structure)
            return orthogonal_structure

        except Exception as e1:
            try:
                from pymatgen.transformations.standard_transformations import SupercellTransformation

                supercell_matrices = [
                    [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                    [[1, -1, 0], [1, 1, 0], [0, 0, 1]],
                    [[2, -1, 0], [1, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
                    [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
                    [[3, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 3, 0], [0, 0, 1]]
                ]

                best_structure = structure.copy()
                best_deviation = max(angle_deviations)

                for matrix in supercell_matrices:
                    try:
                        sc_transformer = SupercellTransformation(matrix)
                        test_structure = sc_transformer.apply_transformation(structure)

                        if len(test_structure) > max_atoms:
                            continue

                        ase_test = AseAtomsAdaptor.get_atoms(test_structure)
                        test_cell_params = ase_test.get_cell_lengths_and_angles()

                        if len(test_cell_params) >= 6:
                            test_angles = [test_cell_params[3], test_cell_params[4], test_cell_params[5]]
                        else:
                            continue
                        test_deviations = []
                        for test_angle in test_angles:
                            test_deviation = abs(float(test_angle) - 90.0)
                            test_deviations.append(test_deviation)

                        max_test_deviation = max(test_deviations)

                        if max_test_deviation < best_deviation:
                            best_structure = test_structure
                            best_deviation = max_test_deviation
                        if max_test_deviation < 0.5:
                            return test_structure

                    except Exception as e2:
                        continue

                return best_structure

            except Exception as e3:

                return structure.copy()

    except Exception as e:

        print(f"Warning: Orthogonal cell search failed: {e}")
        return structure.copy()


def visualize_surface_structure(atoms, title="Surface Structure"):
    try:
        xyz_io = StringIO()
        write(xyz_io, atoms, format="xyz")
        xyz_str = xyz_io.getvalue()

        # Define Jmol colors
        jmol_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
            'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5',
            'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',
            'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
            'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0',
            'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929'
        }

        def add_box(view, cell, color='black', linewidth=2):
            a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
            corners = []
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        corner = i * a + j * b + k * c
                        corners.append(corner)
            edges = []
            for idx in range(8):
                i = idx & 1
                j = (idx >> 1) & 1
                k = (idx >> 2) & 1
                if i == 0:
                    edges.append((corners[idx], corners[idx + 1]))
                if j == 0:
                    edges.append((corners[idx], corners[idx + 2]))
                if k == 0:
                    edges.append((corners[idx], corners[idx + 4]))

            for start, end in edges:
                view.addLine({
                    'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                    'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                    'color': color,
                    'linewidth': linewidth
                })

            arrow_radius = 0.04
            arrow_color = '#000000'
            for vec in [a, b, c]:
                view.addArrow({
                    'start': {'x': 0, 'y': 0, 'z': 0},
                    'end': {'x': float(vec[0]), 'y': float(vec[1]), 'z': float(vec[2])},
                    'color': arrow_color,
                    'radius': arrow_radius
                })
            offset = 0.3

            def add_axis_label(vec, label_val):
                norm = np.linalg.norm(vec)
                end = vec + offset * vec / (norm + 1e-6)
                view.addLabel(label_val, {
                    'position': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                    'fontSize': 14,
                    'fontColor': color,
                    'showBackground': False
                })

            a_len = np.linalg.norm(a)
            b_len = np.linalg.norm(b)
            c_len = np.linalg.norm(c)
            add_axis_label(a, f"a = {a_len:.3f} Å")
            add_axis_label(b, f"b = {b_len:.3f} Å")
            add_axis_label(c, f"c = {c_len:.3f} Å")

        def add_lattice_vectors(view, cell, origin=None):
            if origin is None:
                positions = atoms.get_positions()
                origin = np.mean(positions, axis=0)

            scale = 0.7
            arrow_radius = 0.15

            a_end = origin + cell[0] * scale
            view.addCylinder({
                'start': {'x': origin[0], 'y': origin[1], 'z': origin[2]},
                'end': {'x': a_end[0], 'y': a_end[1], 'z': a_end[2]},
                'radius': arrow_radius,
                'color': 'red'
            })

            view.addSphere({
                'center': {'x': a_end[0], 'y': a_end[1], 'z': a_end[2]},
                'radius': arrow_radius * 2,
                'color': 'red'
            })

            # b-vector (green)
            b_end = origin + cell[1] * scale
            view.addCylinder({
                'start': {'x': origin[0], 'y': origin[1], 'z': origin[2]},
                'end': {'x': b_end[0], 'y': b_end[1], 'z': b_end[2]},
                'radius': arrow_radius,
                'color': 'green'
            })
            # Arrow head for b
            view.addSphere({
                'center': {'x': b_end[0], 'y': b_end[1], 'z': b_end[2]},
                'radius': arrow_radius * 2,
                'color': 'green'
            })

            # c-vector (blue)
            c_end = origin + cell[2] * scale
            view.addCylinder({
                'start': {'x': origin[0], 'y': origin[1], 'z': origin[2]},
                'end': {'x': c_end[0], 'y': c_end[1], 'z': c_end[2]},
                'radius': arrow_radius,
                'color': 'blue'
            })
            # Arrow head for c
            view.addSphere({
                'center': {'x': c_end[0], 'y': c_end[1], 'z': c_end[2]},
                'radius': arrow_radius * 2,
                'color': 'blue'
            })

        # Create 3D view
        view = py3Dmol.view(width=700, height=500)
        view.addModel(xyz_str, "xyz")
        view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})

        # Add unit cell box with lattice vectors and labels
        cell = atoms.get_cell()
        add_box(view, cell, color='black', linewidth=2)

        view.zoomTo()
        view.zoom(1.0)

        html_string = view._make_html()
        components.html(html_string, height=520, width=720)

        unique_elements = sorted(set(atoms.get_chemical_symbols()))

        legend_html = "<div style='display: flex; flex-wrap: wrap; align-items: center; justify-content: center; margin-top: 10px;'>"
        legend_html += "<div style='margin-right: 30px; font-weight: bold; color: #333;'>Elements:</div>"
        for elem in unique_elements:
            color = jmol_colors.get(elem, "#CCCCCC")
            legend_html += (
                f"<div style='margin-right: 15px; display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black; border-radius: 50%;'></div>"
                f"<span style='font-weight: bold;'>{elem}</span></div>"
            )
        legend_html += "</div>"


        st.markdown(legend_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error visualizing structure: {e}")


def generate_surface_slab(atoms, miller_indices, layers, vacuum, supercell, fix_thickness):
    try:
        surface_atoms = surface(atoms, miller_indices, layers, vacuum)
        if supercell != (1, 1, 1):
            surface_atoms = surface_atoms.repeat(supercell)

        positions = surface_atoms.get_positions()
        symbols = surface_atoms.get_chemical_symbols()

        min_z = np.min(positions[:, 2])
        max_z = np.max(positions[:, 2])
        arg_min_z = np.argmin(positions[:, 2])
        arg_max_z = np.argmax(positions[:, 2])
        bottom_element = symbols[arg_min_z]
        top_element = symbols[arg_max_z]

        if fix_thickness > 0:
            fixed_indices = [i for i, pos in enumerate(positions) if pos[2] < min_z + fix_thickness]
            if fixed_indices:
                constraint = FixAtoms(indices=fixed_indices)
                surface_atoms.set_constraint(constraint)

        surface_info = {
            'total_atoms': len(surface_atoms),
            'min_z': min_z,
            'max_z': max_z,
            'bottom_element': bottom_element,
            'top_element': top_element,
            'thickness': max_z - min_z,
            'fixed_atoms': len(fixed_indices) if fix_thickness > 0 else 0,
            'unique_elements': sorted(set(symbols))
        }

        return surface_atoms, surface_info

    except Exception as e:
        st.error(f"Error generating surface: {e}")
        raise e


def generate_lammps_content(atoms, units="metal", atom_style="atomic"):
    """Generate LAMMPS data file with specified units and atom style"""
    try:
        out = StringIO()
        write(out, atoms, format="lammps-data", atom_style=atom_style, units=units)
        return out.getvalue()
    except Exception as e:
        out = StringIO()
        write(out, atoms, format="lammps-data")
        return out.getvalue()


def generate_poscar_content(atoms, use_constraints=False, direct_coords=True, sort_atoms=True,
                            comment="Generated structure"):
    try:
        atoms_copy = atoms.copy()
        if use_constraints and hasattr(atoms, 'constraints') and atoms.constraints:
            return generate_file_content(atoms_copy, "POSCAR", use_constraints=True)

        structure = ase_to_pymatgen(atoms_copy)

        if sort_atoms:
            try:
                structure = structure.get_reduced_structure()
            except:
                pass

        from pymatgen.io.vasp import Poscar
        poscar = Poscar(structure)
        poscar_str = str(poscar)

        lines = poscar_str.split('\n')
        lines[0] = comment

        return '\n'.join(lines)

    except Exception as e:
        out = StringIO()
        write(out, atoms, format="vasp", direct=direct_coords, sort=sort_atoms)
        return out.getvalue()


def generate_xyz_content(atoms, comment="Generated structure"):
    try:
        lines = [str(len(atoms)), comment]

        for atom in atoms:
            pos = atom.position
            lines.append(f"{atom.symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

        return '\n'.join(lines) + '\n'

    except Exception as e:
        out = StringIO()
        write(out, atoms, format="xyz")
        content = out.getvalue()

        # Replace comment line
        lines = content.split('\n')
        if len(lines) > 1:
            lines[1] = comment
            return '\n'.join(lines)
        return content


def generate_cif_content(atoms, title="Generated structure"):
    try:
        structure = ase_to_pymatgen(atoms)
        try:
            structure = structure.get_reduced_structure()
        except:
            pass

        cif_writer = CifWriter(structure)
        cif_content = cif_writer.__str__()

        lines = cif_content.split('\n')
        title_added = False
        for i, line in enumerate(lines):
            if line.startswith('_chemical_name_systematic'):
                lines[i] = f"_chemical_name_systematic           '{title}'"
                title_added = True
                break
            elif line.startswith('data_'):
                lines[i] = f"data_{title.replace(' ', '_')}"

        if not title_added:
            for i, line in enumerate(lines):
                if line.startswith('data_'):
                    lines.insert(i + 1, f"_chemical_name_systematic           '{title}'")
                    break

        return '\n'.join(lines)

    except Exception as e:
        structure = ase_to_pymatgen(atoms)
        cif_writer = CifWriter(structure)
        return cif_writer.__str__()


def generate_file_content(atoms, file_format, use_constraints=True):
    try:
        if file_format == "CIF":
            structure = ase_to_pymatgen(atoms)
            cif_writer = CifWriter(structure)
            return cif_writer.__str__()

        elif file_format == "POSCAR":
            atoms_for_poscar = atoms.copy()
            if use_constraints and hasattr(atoms, 'constraints') and atoms.constraints:
                try:
                    fixed_indices = set()
                    for constraint in atoms.constraints:
                        if hasattr(constraint, 'get_indices'):
                            fixed_indices.update(constraint.get_indices())

                    if fixed_indices:
                        out = StringIO()
                        symbols = atoms_for_poscar.get_chemical_symbols()
                        positions = atoms_for_poscar.get_positions()
                        cell = atoms_for_poscar.get_cell()
                        out.write("Generated surface slab with selective dynamics\n")
                        out.write("1.0\n")

                        for i in range(3):
                            out.write(f"  {cell[i][0]:16.8f}  {cell[i][1]:16.8f}  {cell[i][2]:16.8f}\n")

                        from collections import OrderedDict
                        element_counts = OrderedDict()
                        for symbol in symbols:
                            if symbol in element_counts:
                                element_counts[symbol] += 1
                            else:
                                element_counts[symbol] = 1
                        element_names = list(element_counts.keys())
                        counts = list(element_counts.values())

                        out.write("  " + "  ".join(element_names) + "\n")
                        out.write("  " + "  ".join(map(str, counts)) + "\n")

                        out.write("Selective dynamics\n")
                        out.write("Direct\n")

                        inv_cell = np.linalg.inv(cell)
                        direct_positions = np.dot(positions, inv_cell)

                        for element in element_names:
                            element_indices = [i for i, sym in enumerate(symbols) if sym == element]

                            for atom_idx in element_indices:
                                pos = direct_positions[atom_idx]

                                if atom_idx in fixed_indices:
                                    flags = "F   F   F"
                                else:
                                    flags = "T   T   T"

                                out.write(f"  {pos[0]:16.8f}  {pos[1]:16.8f}  {pos[2]:16.8f}   {flags}\n")

                        return out.getvalue()

                except Exception as e:
                    print(f"Selective dynamics failed: {e}, falling back to regular POSCAR")

            try:
                structure = ase_to_pymatgen(atoms_for_poscar)
                try:
                    structure_reduced = structure.get_reduced_structure()
                    atoms_test = pymatgen_to_ase(structure_reduced)
                    if len(atoms_test) != len(atoms_for_poscar):
                        structure_to_use = structure
                    else:
                        structure_to_use = structure_reduced
                except:
                    structure_to_use = structure

                from pymatgen.io.vasp import Poscar
                poscar = Poscar(structure_to_use)
                return str(poscar)

            except Exception as e:
                try:
                    out = StringIO()
                    write(out, atoms_for_poscar, format="vasp", direct=True, sort=False)
                    return out.getvalue()
                except Exception as e2:
                    return f"Error generating POSCAR: {str(e2)}"

        elif file_format == "XYZ":
            out = StringIO()
            write(out, atoms, format="xyz")
            return out.getvalue()

        elif file_format == "LAMMPS":
            out = StringIO()
            write(out, atoms, format="lammps-data", atom_style="atomic", units="metal")
            return out.getvalue()

        else:
            return f"Unsupported format: {file_format}"

    except Exception as e:
        return f"Error generating {file_format}: {str(e)}"


def combine_structures_vertically(structure1, structure2, vacuum_gap=3.0, match_lattice=True):
    from pymatgen.core import Structure, Lattice
    import numpy as np

    try:
        def remove_vacuum_z(structure, vacuum_threshold=1.0):
            cart_coords = np.array([site.coords for site in structure])

            if len(cart_coords) == 0:
                return structure
            z_coords = cart_coords[:, 2]
            min_z = np.min(z_coords)
            max_z = np.max(z_coords)
            occupied_thickness = max_z - min_z

            buffer = 0.5
            new_c = occupied_thickness + 2 * buffer

            original_c = structure.lattice.c
            if new_c < original_c - vacuum_threshold:
                old_lattice = structure.lattice
                new_lattice = Lattice.from_parameters(
                    a=old_lattice.a, b=old_lattice.b, c=new_c,
                    alpha=old_lattice.alpha, beta=old_lattice.beta, gamma=old_lattice.gamma
                )

                new_species = []
                new_coords = []

                for site in structure:
                    new_z = site.coords[2] - min_z + buffer
                    new_cart_coords = [site.coords[0], site.coords[1], new_z]

                    new_frac_coords = new_lattice.get_fractional_coords(new_cart_coords)

                    new_species.append(site.specie)
                    new_coords.append(new_frac_coords)

                return Structure(new_lattice, new_species, new_coords, coords_are_cartesian=False)
            else:
                return structure

        clean_structure1 = remove_vacuum_z(structure1)
        clean_structure2 = remove_vacuum_z(structure2)

        lat1 = clean_structure1.lattice
        lat2 = clean_structure2.lattice


        if match_lattice:
            a_new = max(lat1.a, lat2.a)
            b_new = max(lat1.b, lat2.b)

            if lat1.a * lat1.b >= lat2.a * lat2.b:
                alpha_new = lat1.alpha
                beta_new = lat1.beta
                gamma_new = lat1.gamma
            else:
                alpha_new = lat2.alpha
                beta_new = lat2.beta
                gamma_new = lat2.gamma
        else:
            a_new = lat1.a
            b_new = lat1.b
            alpha_new = lat1.alpha
            beta_new = lat1.beta
            gamma_new = lat1.gamma

        c_new = lat1.c + lat2.c + vacuum_gap

        new_lattice = Lattice.from_parameters(
            a=a_new, b=b_new, c=c_new,
            alpha=alpha_new, beta=beta_new, gamma=gamma_new
        )

        combined_species = []
        combined_coords = []

        for site in clean_structure1:
            cart_coords = site.coords
            if match_lattice and (lat1.a != a_new or lat1.b != b_new):
                old_frac = clean_structure1.lattice.get_fractional_coords(cart_coords)
                new_cart_coords = [
                    old_frac[0] * a_new,
                    old_frac[1] * b_new,
                    cart_coords[2]
                ]
            else:
                new_cart_coords = cart_coords

            frac_coords = new_lattice.get_fractional_coords(new_cart_coords)

            combined_species.append(site.specie)
            combined_coords.append(frac_coords)

        z_shift = (lat1.c + vacuum_gap) / c_new

        for site in clean_structure2:

            cart_coords = site.coords
            if match_lattice and (lat2.a != a_new or lat2.b != b_new):
                old_frac = clean_structure2.lattice.get_fractional_coords(cart_coords)

                scaled_cart = [
                    old_frac[0] * a_new,
                    old_frac[1] * b_new,
                    cart_coords[2]
                ]
            else:
                scaled_cart = cart_coords
            frac_coords = new_lattice.get_fractional_coords(scaled_cart)

            frac_coords[2] = frac_coords[2] * lat2.c / c_new + z_shift

            combined_species.append(site.specie)
            combined_coords.append(frac_coords)

        combined_structure = Structure(
            lattice=new_lattice,
            species=combined_species,
            coords=combined_coords,
            coords_are_cartesian=False
        )

        return combined_structure

    except Exception as e:
        raise Exception(f"Error combining structures: {str(e)}")


def render_surface_module():

    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #ff6b35; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    st.title("🏗️ Surface Slab Generator")

    st.info("""
        Generate crystal surface slabs from bulk structures using ASE (Atomic Simulation Environment).
        This tool creates surface slabs with specified Miller indices, number of layers, vacuum space, supercell dimensions,
        and fixing the atomic coordinations for the bottom part in POSCAR format.
         You can also combine two structures vertically to create layered systems.
    """)

    col_toggle1, col_toggle2 = st.columns([1, 3])
    with col_toggle1:
        enable_union = st.checkbox(
            "🔗 Structure Union Mode",
            value=False,
            help="Enable to combine two structures vertically into a layered system"
        )

    with col_toggle2:
        if enable_union:
            st.info(
                "💡 **Union Mode**: Combine two structures by placing one on top of the other with automatic lattice matching.")

    if 'full_structures' not in st.session_state or not st.session_state['full_structures']:
        st.warning("Please upload at least one structure file to use the Surface Slab Generator.")
        structure_upload_help = """
        ### How to Use:
        1. **Upload Structure Files**: Use the file uploader in the sidebar to upload your crystal structure files (CIF, POSCAR, LMP, XYZ (with lattice))
        """

        if enable_union:
            structure_upload_help += """
        2. **Structure Union Mode**: Upload at least 2 structures to combine them vertically
        3. **Configure Union**: Set vacuum gap and lattice matching options
        4. **Generate Combined Structure**: Create the layered system
        """
        else:
            structure_upload_help += """
        2. **Select Structure**: Choose which uploaded structure to use for slab generation
        3. **Set Parameters**: Configure Miller indices, number of layers, vacuum space, and other parameters
        4. **Generate Slab**: Create and visualize your surface slab
        """

        structure_upload_help += """
        5. **Download**: Export your slab in various formats (CIF, POSCAR, XYZ, LAMMPS)

        ### Supported Formats:
        - **Input**: CIF, POSCAR, VASP, XSF, LAMMPS, XYZ (with lattice)
        - **Output**: CIF, POSCAR, XYZ, LAMMPS
        """
        st.markdown(structure_upload_help)
        return
    if enable_union:
        file_options = list(st.session_state['full_structures'].keys())

        if len(file_options) < 2:
            st.warning("⚠️ Structure Union Mode requires at least 2 uploaded structures.")
            st.info("Please upload more structure files using the sidebar file uploader.")
            return

        st.markdown(
            """
            <hr style="border: none; height: 3px; background-color: #4CAF50; border-radius: 8px; margin: 20px 0;">
            """,
            unsafe_allow_html=True
        )

        st.subheader("🔗 Structure Union Configuration")

        col_union1, col_union2 = st.columns(2)

        with col_union1:
            st.write("**Bottom Structure (Substrate)**")
            bottom_structure_file = st.selectbox(
                "Select bottom structure:",
                file_options,
                key="bottom_structure_selector"
            )

            bottom_structure = st.session_state['full_structures'][bottom_structure_file]
            st.write(f"• Formula: {bottom_structure.composition.reduced_formula}")
            st.write(f"• Atoms: {len(bottom_structure)}")

            bottom_lattice = bottom_structure.lattice
            st.write(f"• a = {bottom_lattice.a:.3f} Å, b = {bottom_lattice.b:.3f} Å, c = {bottom_lattice.c:.3f} Å")

        with col_union2:
            st.write("**Top Structure (Overlayer)**")
            top_structure_file = st.selectbox(
                "Select top structure:",
                [f for f in file_options if f != bottom_structure_file],
                key="top_structure_selector"
            )

            top_structure = st.session_state['full_structures'][top_structure_file]
            st.write(f"• Formula: {top_structure.composition.reduced_formula}")
            st.write(f"• Atoms: {len(top_structure)}")

            top_lattice = top_structure.lattice
            st.write(f"• a = {top_lattice.a:.3f} Å, b = {top_lattice.b:.3f} Å, c = {top_lattice.c:.3f} Å")

        # Union parameters
        st.write("**Union Parameters**")
        col_param1, col_param2, col_param3 = st.columns(3)

        with col_param1:
            vacuum_gap = st.number_input(
                "Vacuum gap between structures (Å):",
                value=3.0,
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                help="Distance between the top of bottom structure and bottom of top structure"
            )

        with col_param2:
            match_lattice = st.checkbox(
                "Match lattice parameters",
                value=True,
                help="Use the larger a,b lattice parameters to fit both structures"
            )

        with col_param3:
            union_output_name = st.text_input(
                "Output name:",
                value=f"union_{bottom_structure.composition.reduced_formula}_{top_structure.composition.reduced_formula}",
                help="Base name for the combined structure files"
            )

        # Preview combined parameters
        if match_lattice:
            combined_a = max(bottom_lattice.a, top_lattice.a)
            combined_b = max(bottom_lattice.b, top_lattice.b)
            combined_c = bottom_lattice.c + top_lattice.c + vacuum_gap

            st.write("**Predicted Combined Structure:**")
            st.write(f"• Total atoms: {len(bottom_structure) + len(top_structure)}")
            st.write(f"• New lattice: a = {combined_a:.3f} Å, b = {combined_b:.3f} Å, c = {combined_c:.3f} Å")
            st.write(f"• Volume: {combined_a * combined_b * combined_c:.1f} Å³")

        if st.button("🔗 Generate Combined Structure", type="primary"):
            with st.spinner("Combining structures..."):
                try:
                    combined_structure = combine_structures_vertically(
                        bottom_structure,
                        top_structure,
                        vacuum_gap=vacuum_gap,
                        match_lattice=match_lattice
                    )

                    # Store combined structure result
                    st.session_state.union_result = {
                        'structure': combined_structure,
                        'bottom_file': bottom_structure_file,
                        'top_file': top_structure_file,
                        'vacuum_gap': vacuum_gap,
                        'match_lattice': match_lattice,
                        'output_name': union_output_name
                    }

                    st.success("✅ Structures combined successfully!")

                except Exception as e:
                    st.error(f"❌ Error combining structures: {e}")

        if hasattr(st.session_state, 'union_result') and st.session_state.union_result:
            union_result = st.session_state.union_result
            combined_structure = union_result['structure']

            st.markdown(
                """
                <hr style="border: none; height: 2px; background-color: #4CAF50; border-radius: 4px; margin: 15px 0;">
                """,
                unsafe_allow_html=True
            )

            st.subheader("🎯 Combined Structure Results")

            col_result1, col_result2 = st.columns(2)

            with col_result1:
                st.write("**Combined Structure Properties:**")

                result_data = [
                    ["Bottom Structure", union_result['bottom_file']],
                    ["Top Structure", union_result['top_file']],
                    ["Total Atoms", len(combined_structure)],
                    ["Formula", combined_structure.composition.reduced_formula],
                    ["Vacuum Gap", f"{union_result['vacuum_gap']:.1f} Å"],
                    ["Lattice Matched", "Yes" if union_result['match_lattice'] else "No"]
                ]

                result_df = pd.DataFrame(result_data, columns=["Property", "Value"])
                st.dataframe(result_df, use_container_width=True, hide_index=True)

            with col_result2:
                st.write("**Final Lattice Parameters:**")
                final_lattice = combined_structure.lattice

                lattice_data = [
                    ["a (Å)", f"{final_lattice.a:.4f}"],
                    ["b (Å)", f"{final_lattice.b:.4f}"],
                    ["c (Å)", f"{final_lattice.c:.4f}"],
                    ["α (°)", f"{final_lattice.alpha:.2f}"],
                    ["β (°)", f"{final_lattice.beta:.2f}"],
                    ["γ (°)", f"{final_lattice.gamma:.2f}"],
                    ["Volume (Å³)", f"{final_lattice.volume:.1f}"]
                ]

                lattice_df = pd.DataFrame(lattice_data, columns=["Parameter", "Value"])
                st.dataframe(lattice_df, use_container_width=True, hide_index=True)

            combined_atoms = pymatgen_to_ase(combined_structure)
            visualize_surface_structure(combined_atoms, "Combined Layered Structure")

            st.subheader("📥 Download Combined Structure")

            col_union_dl1, col_union_dl2 = st.columns([1, 2])

            with col_union_dl1:
                union_format = st.selectbox(
                    "Download format:",
                    ["CIF", "POSCAR", "XYZ", "LAMMPS"],
                    index=0,
                    key="union_format"
                )

                if union_format == "POSCAR":
                    union_use_constraints = st.checkbox(
                        "Include selective dynamics",
                        value=False,
                        key="union_constraints",
                        help="Add selective dynamics to POSCAR (not applicable for union structures)"
                    )

                    union_direct_coords = st.checkbox(
                        "Use direct coordinates",
                        value=True,
                        key="union_direct",
                        help="Use fractional coordinates instead of cartesian"
                    )

                    union_sort_atoms = st.checkbox(
                        "Sort atoms by element",
                        value=True,
                        key="union_sort",
                        help="Group atoms by element type in POSCAR"
                    )

                elif union_format == "LAMMPS":
                    union_lammps_units = st.selectbox(
                        "LAMMPS units:",
                        ["metal", "real", "si", "cgs", "electron", "micro", "nano"],
                        index=0,
                        key="union_lammps_units",
                        help="Unit system for LAMMPS data file"
                    )

                    union_atom_style = st.selectbox(
                        "Atom style:",
                        ["atomic", "charge", "molecular", "full"],
                        index=0,
                        key="union_atom_style",
                        help="LAMMPS atom style for data file"
                    )

                    union_masses = st.checkbox(
                        "Include masses",
                        value=True,
                        key="union_masses",
                        help="Include atomic masses in LAMMPS file"
                    )

                    union_force_skew = st.checkbox(
                        "Force skew",
                        value=False,
                        key="union_force_skew",
                        help="Force skewed cell format for non-orthogonal cells"
                    )

                elif union_format == "XYZ":
                    union_xyz_comment = st.text_input(
                        "Comment line:",
                        value=f"Combined structure: {union_result['bottom_file']} + {union_result['top_file']}",
                        key="union_xyz_comment",
                        help="Comment line for XYZ file header"
                    )

                    union_extended_xyz = st.checkbox(
                        "Extended XYZ format",
                        value=True,
                        key="union_extended_xyz",
                        help="Include lattice information in XYZ file"
                    )

                elif union_format == "CIF":
                    union_cif_title = st.text_input(
                        "Structure title:",
                        value=f"Combined_{combined_structure.composition.reduced_formula}",
                        key="union_cif_title",
                        help="Title for the CIF file"
                    )

                    union_symprec = st.number_input(
                        "Symmetry precision:",
                        value=0.01,
                        min_value=0.001,
                        max_value=0.1,
                        step=0.001,
                        format="%.3f",
                        key="union_symprec",
                        help="Precision for symmetry detection"
                    )

            with col_union_dl2:
                if st.button(f"🔧 Prepare Combined {union_format} File", key="prepare_union_file", type="secondary"):
                    with st.spinner(f"Preparing {union_format} file..."):
                        try:
                            # Generate file content with appropriate options
                            if union_format == "LAMMPS":
                                from ase.io import write
                                from io import StringIO

                                sio = StringIO()
                                write(sio, combined_atoms, format="lammps-data",
                                      atom_style=union_atom_style,
                                      units=union_lammps_units,
                                      masses=union_masses,
                                      force_skew=union_force_skew)
                                union_file_content = sio.getvalue()

                            elif union_format == "POSCAR":
                                union_file_content = generate_poscar_content(
                                    combined_atoms,
                                    use_constraints=union_use_constraints,
                                    direct_coords=union_direct_coords,
                                    sort_atoms=union_sort_atoms,
                                    comment=f"Combined structure: {union_result['bottom_file']} + {union_result['top_file']}"
                                )

                            elif union_format == "XYZ":
                                if union_extended_xyz:
                                    lattice_vectors = combined_structure.lattice.matrix
                                    cart_coords = []
                                    elements = []
                                    for site in combined_structure:
                                        cart_coords.append(
                                            combined_structure.lattice.get_cartesian_coords(site.frac_coords))
                                        elements.append(site.specie.symbol)

                                    xyz_lines = [str(len(combined_structure))]
                                    lattice_string = " ".join([f"{x:.6f}" for row in lattice_vectors for x in row])
                                    properties = "Properties=species:S:1:pos:R:3"
                                    xyz_lines.append(f'Lattice="{lattice_string}" {properties} {union_xyz_comment}')

                                    for element, coord in zip(elements, cart_coords):
                                        xyz_lines.append(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

                                    union_file_content = "\n".join(xyz_lines)
                                else:
                                    union_file_content = generate_xyz_content(
                                        combined_atoms,
                                        comment=union_xyz_comment
                                    )

                            elif union_format == "CIF":
                                from pymatgen.io.cif import CifWriter
                                cif_writer = CifWriter(combined_structure, symprec=union_symprec, refine_struct=False)
                                cif_content = str(cif_writer)

                                lines = cif_content.split('\n')
                                title_added = False
                                for i, line in enumerate(lines):
                                    if line.startswith('_chemical_name_systematic'):
                                        lines[i] = f"_chemical_name_systematic           '{union_cif_title}'"
                                        title_added = True
                                        break
                                    elif line.startswith('data_'):
                                        lines[i] = f"data_{union_cif_title.replace(' ', '_')}"

                                if not title_added:
                                    for i, line in enumerate(lines):
                                        if line.startswith('data_'):
                                            lines.insert(i + 1,
                                                         f"_chemical_name_systematic           '{union_cif_title}'")
                                            break

                                union_file_content = '\n'.join(lines)

                            union_file_extension = {
                                "CIF": "cif",
                                "POSCAR": "poscar",
                                "XYZ": "xyz",
                                "LAMMPS": "lmp"
                            }[union_format]

                            union_filename = f"{union_result['output_name']}.{union_file_extension}"
                            st.session_state.prepared_union_file = {
                                'content': union_file_content,
                                'filename': union_filename,
                                'format': union_format,
                                'options': {
                                    'poscar_direct': union_direct_coords if union_format == "POSCAR" else None,
                                    'poscar_sort': union_sort_atoms if union_format == "POSCAR" else None,
                                    'poscar_constraints': union_use_constraints if union_format == "POSCAR" else None,
                                    'lammps_units': union_lammps_units if union_format == "LAMMPS" else None,
                                    'lammps_style': union_atom_style if union_format == "LAMMPS" else None,
                                    'lammps_masses': union_masses if union_format == "LAMMPS" else None,
                                    'lammps_skew': union_force_skew if union_format == "LAMMPS" else None,
                                    'xyz_comment': union_xyz_comment if union_format == "XYZ" else None,
                                    'xyz_extended': union_extended_xyz if union_format == "XYZ" else None,
                                    'cif_title': union_cif_title if union_format == "CIF" else None,
                                    'cif_symprec': union_symprec if union_format == "CIF" else None,
                                },
                                'file_size': len(union_file_content),
                                'atom_count': len(combined_atoms)
                            }

                            st.success(f"✅ Combined {union_format} file prepared successfully!")

                        except Exception as e:
                            st.error(f"❌ Error preparing {union_format} file: {e}")
                            st.info("💡 Try adjusting the format options or use a simpler format.")

                if hasattr(st.session_state, 'prepared_union_file') and st.session_state.prepared_union_file:
                    prepared_union = st.session_state.prepared_union_file
                    if prepared_union['format'] == union_format:
                        st.download_button(
                            label=f"📥 Download Combined {prepared_union['format']} File",
                            data=prepared_union['content'],
                            file_name=prepared_union['filename'],
                            mime="text/plain",
                            type="primary",
                            help=f"Download the prepared {prepared_union['format']} file"
                        )


                        st.info(
                            f"📄 File ready: {prepared_union['filename']} ({prepared_union['file_size']:,} bytes, {prepared_union['atom_count']:,} atoms)")

        return

    file_options = list(st.session_state['full_structures'].keys())
    selected_file = st.sidebar.selectbox(
        "Select structure for surface slab generation:",
        file_options,
        key="surface_structure_selector"
    )

    try:
        structure = st.session_state['full_structures'][selected_file]
        atoms = pymatgen_to_ase(structure)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Structure Information")
            try:
                analyzer = SpacegroupAnalyzer(structure)
                spg_symbol = analyzer.get_space_group_symbol()
                spg_number = analyzer.get_space_group_number()
                st.write(f"**Space group:** {spg_symbol} (#{spg_number})")
            except:
                st.write("**Space group:** Could not determine")

            lattice = structure.lattice
            st.write("**Lattice parameters:**")
            st.write(f"a = {lattice.a:.4f} Å, b = {lattice.b:.4f} Å, c = {lattice.c:.4f} Å")
            st.write(f"α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")

            comp = structure.composition
            st.write("**Composition:**")
            comp_data = []
            for el, amt in comp.items():
                comp_data.append({
                    "Element": el.symbol,
                    "Count": int(amt),
                    "Fraction": f"{amt / comp.num_atoms:.3f}"
                })
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)

        with col2:
            st.subheader("🔍 Original Structure Preview")
            original_atoms = pymatgen_to_ase(structure)
            visualize_surface_structure(original_atoms, "Original Bulk Structure")

        st.markdown(
            """
            <hr style="border: none; height: 3px; background-color: #ff6b35; border-radius: 8px; margin: 20px 0;">
            """,
            unsafe_allow_html=True
        )

        st.subheader("⚙️ Surface Slab Parameters")

        col_param1, col_param2, col_param3 = st.columns(3)

        with col_param1:
            st.write("**Miller Indices**")
            miller_h = st.number_input("h index:", value=1, min_value=-10, max_value=10, step=1, key="miller_h")
            miller_k = st.number_input("k index:", value=0, min_value=-10, max_value=10, step=1, key="miller_k")
            miller_l = st.number_input("l index:", value=0, min_value=-10, max_value=10, step=1, key="miller_l")
            miller_indices = (miller_h, miller_k, miller_l)

            st.write("**Surface Properties**")
            layers = st.number_input("Number of layers:", value=10, min_value=3, max_value=50, step=1)
            vacuum = st.number_input("Vacuum space (Å):", value=15.0, min_value=0.0, max_value=50.0, step=0.5)

        with col_param2:
            st.write("**Supercell**")
            nx = st.number_input("x-direction:", value=1, min_value=1, max_value=50, step=1, key="surf_nx")
            ny = st.number_input("y-direction:", value=1, min_value=1, max_value=50, step=1, key="surf_ny")
            nz = st.number_input("z-direction:", value=1, min_value=1, max_value=30, step=1, key="surf_nz")
            supercell = (nx, ny, nz)

            st.write("**Constraints**")
            fix_thickness = st.number_input(
                "Fix bottom layers (Å):",
                value=4.0,
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                help="Atoms within this distance from the bottom will be fixed"
            )

        with col_param3:
            st.write("**Output Options**")
            output_format = st.selectbox(
                "File format:",
                ["CIF", "POSCAR", "XYZ", "LAMMPS"],
                index=0
            )

            include_constraints = st.checkbox(
                "Include constraints in POSCAR",
                value=True,
                help="Add selective dynamics to POSCAR for fixed atoms"
            )

            output_name = st.text_input(
                "Output filename:",
                value=f"surface_{miller_h}{miller_k}{miller_l}",
                help="Filename without extension"
            )

        st.write("**Preview Parameters:**")
        st.write(f"• Miller indices: ({miller_h} {miller_k} {miller_l})")
        st.write(f"• Supercell dimensions: {nx} × {ny} × {nz}")
        st.write(f"• Estimated atoms in slab: ~{len(atoms) * layers * nx * ny * nz}")

        if st.button("🚀 Generate Surface Slab", type="primary"):
            with st.spinner("Generating surface slab..."):
                try:
                    surface_atoms, surface_info = generate_surface_slab(
                        atoms, miller_indices, layers, vacuum, supercell, fix_thickness
                    )
                    st.session_state.surface_result = {
                        'atoms': surface_atoms,
                        'info': surface_info,
                        'parameters': {
                            'miller_indices': miller_indices,
                            'layers': layers,
                            'vacuum': vacuum,
                            'supercell': supercell,
                            'fix_thickness': fix_thickness
                        }
                    }

                    st.success("✅ Surface slab generated successfully!")

                except Exception as e:
                    st.error(f"Error generating surface: {e}")

        if hasattr(st.session_state, 'surface_result') and st.session_state.surface_result:
            result = st.session_state.surface_result
            surface_atoms = result['atoms']
            surface_info = result['info']
            parameters = result['parameters']

            st.markdown(
                """
                <hr style="border: none; height: 3px; background-color: #ff6b35; border-radius: 8px; margin: 20px 0;">
                """,
                unsafe_allow_html=True
            )

            st.subheader("🎯 Generated Surface Slab")


            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.write("**Surface Properties:**")
                info_data = [
                    ["Miller Indices",
                     f"({parameters['miller_indices'][0]} {parameters['miller_indices'][1]} {parameters['miller_indices'][2]})"],
                    ["Total Atoms", surface_info['total_atoms']],
                    ["Slab Thickness", f"{surface_info['thickness']:.2f} Å"],
                    ["Bottom Element", surface_info['bottom_element']],
                    ["Top Element", surface_info['top_element']],
                    ["Fixed Atoms", surface_info['fixed_atoms']],
                    ["Vacuum Space", f"{parameters['vacuum']:.1f} Å"],
                    ["Supercell",
                     f"{parameters['supercell'][0]}×{parameters['supercell'][1]}×{parameters['supercell'][2]}"]
                ]

                info_df = pd.DataFrame(info_data, columns=["Property", "Value"])
                st.dataframe(info_df, use_container_width=True, hide_index=True)

            with col_info2:
                st.write("**Element Composition:**")
                elements, counts = np.unique(surface_atoms.get_chemical_symbols(), return_counts=True)
                comp_data = []
                for elem, count in zip(elements, counts):
                    comp_data.append({
                        "Element": elem,
                        "Count": count,
                        "Percentage": f"{100 * count / len(surface_atoms):.1f}%"
                    })
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                st.write("**Lattice Parameters:**")
                slab_lattice = ase_to_pymatgen(surface_atoms).lattice
                lattice_data = [
                    ["a (Å)", f"{slab_lattice.a:.4f}"],
                    ["b (Å)", f"{slab_lattice.b:.4f}"],
                    ["c (Å)", f"{slab_lattice.c:.4f}"],
                    ["α (°)", f"{slab_lattice.alpha:.2f}"],
                    ["β (°)", f"{slab_lattice.beta:.2f}"],
                    ["γ (°)", f"{slab_lattice.gamma:.2f}"]
                ]
                lattice_df = pd.DataFrame(lattice_data, columns=["Parameter", "Value"])
                st.dataframe(lattice_df, use_container_width=True, hide_index=True)

            if len(surface_atoms) > 100000:
                st.warning(
                    f"⚠️ Structure has {len(surface_atoms):,} atoms. 3D visualization disabled for performance reasons.")
                st.info(
                    "💡 You can still download the structure files. For visualization, consider using specialized software like VESTA, OVITO, or VMD.")
            else:
                visualize_surface_structure(surface_atoms, "Generated Surface Slab")

            st.markdown(
                """
                <hr style="border: none; height: 2px; background-color: #ffa500; border-radius: 4px; margin: 15px 0;">
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <hr style="border: none; height: 2px; background-color: #ff6b35; border-radius: 4px; margin: 15px 0;">
                """,
                unsafe_allow_html=True
            )

            st.subheader("📥 Download Original Surface Slab")

            col_dl1, col_dl2 = st.columns([1, 2])

            with col_dl1:
                final_format = st.selectbox(
                    "Download format:",
                    ["CIF", "POSCAR", "XYZ", "LAMMPS"],
                    index=["CIF", "POSCAR", "XYZ", "LAMMPS"].index(output_format),
                    key="original_format"
                )

                if final_format == "POSCAR":
                    use_constraints = st.checkbox(
                        "Include selective dynamics",
                        value=include_constraints,
                        key="download_constraints"
                    )
                else:
                    use_constraints = False

            with col_dl2:
                if st.button(f"🔧 Prepare {final_format} File", key="prepare_original"):
                    try:
                        file_content = generate_file_content(
                            surface_atoms,
                            final_format,
                            use_constraints if final_format == "POSCAR" else False
                        )

                        file_extension = {
                            "CIF": "cif",
                            "POSCAR": "poscar",
                            "XYZ": "xyz",
                            "LAMMPS": "lmp"
                        }[final_format]

                        filename = f"{output_name}_{miller_h}{miller_k}{miller_l}_{layers}L_{vacuum:.0f}vac.{file_extension}"
                        st.session_state.prepared_file = {
                            'content': file_content,
                            'filename': filename,
                            'format': final_format
                        }

                        st.success(f"✅ {final_format} file prepared successfully!")

                    except Exception as e:
                        st.error(f"Error generating {final_format} file: {e}")
                if hasattr(st.session_state, 'prepared_file') and st.session_state.prepared_file:
                    prepared = st.session_state.prepared_file
                    st.download_button(
                        label=f"📥 Download {prepared['format']} File",
                        data=prepared['content'],
                        file_name=prepared['filename'],
                        mime="text/plain",
                        type="primary"
                    )

    except Exception as e:
        st.error(f"Error processing structure: {e}")
        import traceback
        st.error(traceback.format_exc())
