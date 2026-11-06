{
  description = "thinkctl dev shells (no global installs). default: X11 screenshot/input/recording tools. build: CUDA 12.8 toolchain to compile llama.cpp";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };  # unfree: NVIDIA CUDA redistributables
      cuda = pkgs.cudaPackages_12_8;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [ maim scrot xdotool xbindkeys wmctrl xorg.xwininfo xorg.xprop ffmpeg imagemagick ];
        shellHook = ''echo "thinkctl dev shell (X11 tools)"'';
      };
      devShells.x86_64-linux.build = (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
        packages = [ pkgs.cmake pkgs.ninja pkgs.git cuda.cuda_nvcc cuda.cuda_cudart cuda.cuda_cccl cuda.libcublas ];
        shellHook = ''echo "thinkctl build shell (CUDA ${cuda.cudaMajorMinorVersion})"'';
      };
    };
}
