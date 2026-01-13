{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {...} @ inputs: let
    supportedSystems = ["x86_64-linux"];
    forEachSupportedSystem = f:
      inputs.nixpkgs.lib.genAttrs supportedSystems (system: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        python = pkgs.python313;
      in
        f {
          inherit pkgs python;
        });
  in {
    devShells = forEachSupportedSystem ({
      pkgs,
      python,
    }: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          python
          uv
        ];

        env = {
          CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (pkgs.pythonManylinuxPackages.manylinux1
            ++ [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              "/run/opengl-driver"
            ]);

          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = python.interpreter;
        };

        shellHook = ''
          unset PYTHONPATH
        '';
      };
    });
  };
}
