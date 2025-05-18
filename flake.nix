# src: https://tonyfinn.com/blog/nix-from-first-principles-flake-edition/nix-8-flakes-and-developer-environments
{
	inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

	outputs = { self, nixpkgs }:
	let
		system = "x86_64-linux";
		pkgs = import nixpkgs {
			inherit system;
			config.allowUnfree = true;
		};
	in {
		devShells.${system}.default = pkgs.mkShell rec {
			packages = with pkgs; [
				# pkg-config
				# wayland
				# wayland-protocols
				libxkbcommon
				# libinput
				# udev
				# xorg.libX11  # Needed as fallback or for XWayland
				# xorg.libXcursor
				# xorg.libXrandr
				# xorg.libXi
				# vulkan-loader
			];
			# Set environment variables here:
			# MY_ENV_VAR = 1;
			# WINIT_UNIX_BACKEND = "wayland";
			# LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
			# RUST_BACKTRACE = "full";
		};
		# Define extra shells or packages here.
	};
}
