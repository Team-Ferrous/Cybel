file(REMOVE_RECURSE
  "libanvil_runtime_core.pdb"
  "libanvil_runtime_core.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/anvil_runtime_core.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
