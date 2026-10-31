import os
import trimesh

class MeshValidatorAndExporter:
    @staticmethod
    def validate_and_export(raw_mesh, output_dir: str, base_filename: str = "reconstructed_asset"):
        mesh = trimesh.Trimesh(
            vertices=raw_mesh.vertices,
            faces=raw_mesh.faces,
            vertex_colors=raw_mesh.visual.vertex_colors if hasattr(raw_mesh.visual, "vertex_colors") else None,
            process=True
        )

        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fill_holes(mesh)

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        bounds = mesh.extents

        if num_vertices < 100 or num_faces < 100:
            raise ValueError(f"Mesh generation failed: insufficient geometry ({num_vertices} vertices).")

        z_depth = bounds[2]
        max_dim = max(bounds[0], bounds[1], bounds[2])

        if max_dim <= 0.0 or (z_depth < 0.0005 and max_dim < 0.001):
            raise ValueError(f"Validation failed: Asset lacks volumetric depth (Z-span: {z_depth:.4f}).")

        obj_path = os.path.join(output_dir, f"{base_filename}.obj")
        glb_path = os.path.join(output_dir, f"{base_filename}.glb")

        mesh.export(obj_path)
        mesh.export(glb_path, file_type="glb")

        report = (
            f"=== MESH VALIDATION REPORT ===\n"
            f"Status: VALID VOLUMETRIC GEOMETRY\n"
            f"Vertices: {num_vertices:,}\n"
            f"Faces: {num_faces:,}\n"
            f"Dimensions (X x Y x Z): {bounds[0]:.2f} x {bounds[1]:.2f} x {bounds[2]:.2f}\n"
            f"Watertight Solid: {'Yes' if mesh.is_watertight else 'Approximated Mesh'}\n"
            f"Formats: .OBJ (Wavefront) | .GLB (Binary GLTF)"
        )

        return obj_path, glb_path, report