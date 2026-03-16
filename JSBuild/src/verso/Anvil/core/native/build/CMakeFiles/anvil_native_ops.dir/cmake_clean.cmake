file(REMOVE_RECURSE
  "libanvil_native_ops.pdb"
  "libanvil_native_ops.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/anvil_native_ops.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
