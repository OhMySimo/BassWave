# Copy both files from this gist or create them:
python3 /path/to/gen_shim.py  # generates /tmp/hip_compat.c and /tmp/hip_compat.map

# Compile — uses your actual ROCm headers for correct type definitions
gcc -shared -fPIC \
    -I/opt/rocm-7.0.2/include \
    -Wl,--version-script=/tmp/hip_compat.map \
    -o /home/simone/hip_compat.so \
    /tmp/hip_compat.c \
    -ldl

# Verify the versioned symbols are exported
nm -D /home/simone/hip_compat.so | grep "@hip_"
