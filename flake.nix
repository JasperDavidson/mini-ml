{
	description = "mini-ml flake";

	inputs = {
		nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
	};

	outputs = { self, nixpkgs, ... }: let
		pkgs = nixpkgs.legacyPackages."x86_64-linux";
	in {
		devShells.x86_64-linux.default = pkgs.mkShell {

			packages = [
				pkgs.python3
			];

			env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
				pkgs.stdenv.cc.cc.lib
				pkgs.libz
			];
			
		};
	};
}
