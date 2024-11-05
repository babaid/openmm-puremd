//
// Created by babaid on 05.10.24.
//
#include "openmm/internal/PuremdInterface.h"
#include "openmm/OpenMMException.h"

#include "spuremd.h"
#include<iostream>

using namespace OpenMM;
PuremdInterface::PuremdInterface(): firstCall(true) {}


void PuremdInterface::setInputFileNames(const std::string &ffieldFilename, const std::string &controlFilename) {

  ffield_filename = ffieldFilename;
  control_filename = controlFilename;
}

void PuremdInterface::getReaxffPuremdForces(double temperature, int num_qm_atoms,  const std::vector<char> &qm_symbols, const std::vector<double> & qm_pos,
                                            int num_mm_atoms, const  std::vector<char> &mm_symbols, const std::vector<double> & mm_pos_q,
                                            const std::vector<double> & sim_box_info,
                                            std::vector<double>& qm_forces, std::vector<double>& mm_forces, std::vector<double>& qm_q, double& totalEnergy) {

  if(firstCall)
  {
    handlePuremd = setup_qmmm(
          num_qm_atoms, qm_symbols.data(), qm_pos.data(), num_mm_atoms,
          mm_symbols.data(), mm_pos_q.data(), sim_box_info.data(),
          ffield_filename.c_str(), control_filename.c_str());
    firstCall = false;
  }
  else
  {
      retPuremd = reset_qmmm(handlePuremd, num_qm_atoms, qm_symbols.data(),  qm_pos.data(),
                     num_mm_atoms, mm_symbols.data(), mm_pos_q.data(),
                     sim_box_info.data(),
                     ffield_filename.c_str(), control_filename.c_str());
      if(0 != retPuremd) throw OpenMMException("Issue with PuReMD function reset_qmmm.");
  }
  // tweaking with the dissipation energy. The current setup ensures that if the eds 
  // vector is empty, the original forcefield parameters are used automatically

  retPuremd = set_dissipation_energies_to_temperature(handlePuremd, temperature);
  

  retPuremd = simulate(handlePuremd);
  
  if (0 != retPuremd) throw OpenMMException("Error at PuReMD simulation.");
  retPuremd = get_atom_forces_qmmm(handlePuremd, qm_forces.data(), mm_forces.data());
  retPuremd = get_atom_charges_qmmm(handlePuremd, qm_q.data(), NULL);
  retPuremd = get_system_info(handlePuremd, NULL, NULL, &totalEnergy, NULL, NULL, NULL);
  if(0!=retPuremd) throw OpenMMException("Error in parameter extraction.");
  //double be, vdwe, ele, pol;
  //retPuremd = get_energies_qmmm(handlePuremd, &be, &vdwe, &ele, &pol);

//std::cout << "Bonding energy: " << be << " vdW energy: " << vdwe << std::endl << " ele energy: " << ele << " pol en: "<< pol<< std::endl;


}
