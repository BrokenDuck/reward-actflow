def _lazy_import(module_path, class_name):
    class LazySetup:
        _cls = None

        @classmethod
        def _resolve(cls):
            if cls._cls is None:
                import importlib
                mod = importlib.import_module(module_path, package="adm.setups")
                cls._cls = getattr(mod, class_name)
            return cls._cls

        def __new__(cls, args, device=None):
            return cls._resolve()(args, device=device)

        @classmethod
        def add_args(cls, parser):
            return cls._resolve().add_args(parser)

    LazySetup.__name__ = class_name
    LazySetup.__qualname__ = class_name
    return LazySetup


setups = {
    "toy": _lazy_import(".toy", "ToyProblemSetup"),
    "qm9": _lazy_import(".molecules", "QM9ProblemSetup"),
    "geom_drugs": _lazy_import(".molecules", "GEOMDrugsProblemSetup"),
    "stable_diffusion": _lazy_import(".stable_diffusion", "StableDiffusionProblemSetup"),
    "proteins": _lazy_import(".proteins", "ProteinProblemSetup"),
    "peptides": _lazy_import(".peptides", "PeptideProblemSetup"),
}
