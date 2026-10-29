"""External renderer preflight dependency for the bundled evaluator snapshot."""


class Sapien_TEST:
    def __init__(self):
        import sapien.core as sapien
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        sapien.render.set_camera_shader_dir('rt')
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser('oidn')
        self.scene = self.engine.create_scene(sapien.SceneConfig())
        print('Render Well', flush=True)
