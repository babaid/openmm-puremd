set(command "/home/babaid/mambaforge/envs/openmm_build/bin/cmake;-E;copy;/home/babaid/repos/opmd/reaxff_puremd/PuReMD/sPuReMD/lib/.libs/libspuremd.so.1.0.0;/home/babaid/mambaforge/envs/openmm_build/include/openmm/lib/libspuremd.so.1.0.0")

execute_process(COMMAND ${command} RESULT_VARIABLE result)
if(result)
  set(msg "Command failed (${result}):\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  message(FATAL_ERROR "${msg}")
endif()
set(command "/home/babaid/mambaforge/envs/openmm_build/bin/cmake;-E;create_symlink;libspuremd.so.1.0.0;/home/babaid/mambaforge/envs/openmm_build/include/openmm/lib/libspuremd.so.1")

execute_process(COMMAND ${command} RESULT_VARIABLE result)
if(result)
  set(msg "Command failed (${result}):\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  message(FATAL_ERROR "${msg}")
endif()
set(command "/home/babaid/mambaforge/envs/openmm_build/bin/cmake;-E;create_symlink;libspuremd.so.1;/home/babaid/mambaforge/envs/openmm_build/include/openmm/lib/libspuremd.so")

execute_process(COMMAND ${command} RESULT_VARIABLE result)
if(result)
  set(msg "Command failed (${result}):\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  message(FATAL_ERROR "${msg}")
endif()