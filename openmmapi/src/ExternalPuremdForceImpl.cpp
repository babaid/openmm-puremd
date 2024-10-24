//
// Created by babaid on 05.10.24.
//

#include "openmm/internal/ExternalPuremdForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/PuremdInterface.h"
#include "openmm/kernels.h"
#include "openmm/Units.h"
#include <algorithm>
#include <sstream>
#include<iostream>

using namespace OpenMM;
using namespace std;

constexpr double conversionFactor = 1.66053906892 * 6.02214076/100;

inline void transformPosQM(const std::vector<Vec3>& positions, const std::vector<int> indices, std::vector<double> out){
  std::for_each(indices.begin(), indices.end(), [&](int Index){
    out.emplace_back(positions[Index][0]*AngstromsPerNm);
    out.emplace_back(positions[Index][1]*AngstromsPerNm);
    out.emplace_back(positions[Index][2]*AngstromsPerNm);
  });
}

inline void transformPosqMM(const std::vector<Vec3>& positions, const std::vector<double> charges, const std::vector<int> indices, std::vector<double> out){
  std::for_each(indices.begin(), indices.end(), [&](int Index){
    out.emplace_back(positions[Index][0]*AngstromsPerNm);
    out.emplace_back(positions[Index][1]*AngstromsPerNm);
    out.emplace_back(positions[Index][2]*AngstromsPerNm);
    out.emplace_back(charges[Index]);
  });
}

ExternalPuremdForceImpl::ExternalPuremdForceImpl(const ExternalPuremdForce &owner): CustomCPPForceImpl(owner), owner(owner)
{
  std::string ffield_file, control_file;
  owner.getFileNames(ffield_file, control_file);
  Interface.setInputFileNames(ffield_file, control_file);
    for(int i = 0; i<owner.getNumAtoms(); ++i)
    {
      int particle;
      char symbol[2];
      int isqm;
      owner.getParticleParameters(i, particle, symbol, isqm);
      if(isqm)
      {
        qmParticles.emplace_back(particle);
        qmSymbols.emplace_back(symbol[0]);
        qmSymbols.emplace_back(symbol[1]);
      }
      else
      {
        mmParticles.emplace_back(particle);
        mmSymbols.emplace_back(symbol[0]);
        mmSymbols.emplace_back(symbol[1]);
      }
    }
}

/**
This calculates the box size and angles. For now it is not needed because we assume that there are 90 degree angles.
*/
void ExternalPuremdForceImpl::getBoxInfo(ContextImpl& context, std::vector<double>& simBoxInfo)
{
  std::vector<Vec3> PeriodicBoxVectors(3);
  context.getPeriodicBoxVectors(PeriodicBoxVectors[0], PeriodicBoxVectors[1], PeriodicBoxVectors[2]);

  for (int i=0; i<3; i++)
  {
    //AA
    simBoxInfo[i] = std::sqrt(PeriodicBoxVectors[i].dot(PeriodicBoxVectors[i]))*AngstromsPerNm*10;
  }
  simBoxInfo[3] = std::acos(PeriodicBoxVectors[1].dot(PeriodicBoxVectors[2])/(simBoxInfo[1]*simBoxInfo[2])) * 180.0 / M_PI;
  simBoxInfo[4] =  std::acos(PeriodicBoxVectors[0].dot(PeriodicBoxVectors[2])/(simBoxInfo[0]*simBoxInfo[2])) * 180.0 / M_PI;
  simBoxInfo[5] =  std::acos(PeriodicBoxVectors[0].dot(PeriodicBoxVectors[1])/(simBoxInfo[0]*simBoxInfo[1])) * 180.0 / M_PI;
}

double ExternalPuremdForceImpl::computeForce(ContextImpl& context, const std::vector<Vec3> &positions, std::vector<Vec3>& forces)
{
  // need to seperate positions
  //next we need to seperate and flatten the QM/MM positions and convert to AA#
  int N = owner.getNumAtoms();
  int numQm = qmParticles.size(), numMm = mmParticles.size();
  std::vector<double> qmPos, mmPos_q;
  //get the box size. move this into a function
  
  std::vector<double> simBoxInfo(6);
  getBoxInfo(context, simBoxInfo);

  qmPos.reserve(numQm*3);
  mmPos_q.reserve(numMm*3);
  
  transformPosQM(positions, qmParticles, qmPos);

  //retrieve charges from the context. Had to introduce some changes to classes Context, ContextImpl,
  // UpdateStateDataKernel, CommonUpdateStateDataKernel  
  std::vector<double> charges;
  charges.reserve(N);
  context.getCharges(charges);

  transformPosqMM(positions, charges, mmParticles, mmPos_q);

  // OUTPUT VARIABLES
  std::vector<double> qmForces(numQm*3), mmForces(numMm*3);
  std::vector<double> qmQ(numQm, 0);
  double energy;

  Interface.getReaxffPuremdForces(numQm, qmSymbols, qmPos,
                                  numMm, mmSymbols, mmPos_q,
                                  simBoxInfo, qmForces, mmForces, qmQ,
                                  energy);

  // merge the qm and mm forces, additionally transform the scale
  std::vector<Vec3> transformedForces(owner.getNumAtoms());


  for (size_t i=0; i<qmParticles.size(); ++i)
  {
      transformedForces[qmParticles[i]][0] = qmForces.at(3*i);
      transformedForces[qmParticles[i]][1] = qmForces.at(3*i + 1);
      transformedForces[qmParticles[i]][2] = qmForces.at(3*i + 2);
      charges[qmParticles[i]] = qmQ[i];
  }

  for (size_t i=0; i<mmParticles.size(); ++i)
  {
      transformedForces[mmParticles[i]][0] = mmForces[i*3];
      transformedForces[mmParticles[i]][1] = mmForces[i*3 + 1];
      transformedForces[mmParticles[i]][2] = mmForces[i*3 + 2];
  }

  //update charges
  context.setCharges(charges);
  //copy forces and transform from Angstroms * Daltons / ps^2 to kJ/mol/nm

  for(size_t i =0;i<forces.size();++i) {
      forces[i] = -transformedForces[i] * conversionFactor;
  }

  //done
  //kCal -> kJ
  return energy*KJPerKcal;
}
