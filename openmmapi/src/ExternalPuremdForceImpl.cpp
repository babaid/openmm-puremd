//
// Created by babaid on 05.10.24.
//

#include "openmm/internal/ExternalPuremdForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/PuremdInterface.h"
#include "openmm/kernels.h"
#include "openmm/Units.h"
#include "omp.h"
#include<thread>
#include<mutex>
#include <algorithm>
#include <sstream>
#include<iostream>
#include<limits>

using namespace OpenMM;
using namespace std;

constexpr double conversionFactor = (1.66053906892 * 6.02214076)*4.184/2;
constexpr double ProtonToCoulomb = 1.602E-19;

inline void transformPosQM(const std::vector<Vec3>& positions, const std::vector<int> indices, std::vector<double>& charges,  std::vector<double>& out){
    std::for_each(indices.begin(), indices.end(), [&](int Index){
    out.emplace_back(positions[Index][0]*AngstromsPerNm);
    out.emplace_back(positions[Index][1]*AngstromsPerNm);
    out.emplace_back(positions[Index][2]*AngstromsPerNm);
    //This is very important, because otherwise we need to create exceptions for all the QM atoms.
    //For small systems this imposes no problem, but this means N_qm*N_mm exceptions, for large systems CUDA will crash.
    charges[Index] = 0;
  });
}

inline void transformPosqMM(const std::vector<Vec3>& positions, const std::vector<double>& charges, const std::vector<int>& indices, std::vector<double>& out){
    out.reserve(indices.size()*4);
    std::for_each(indices.begin(), indices.end(), [&](int Index){
        out.emplace_back(positions[Index][0]*AngstromsPerNm);
        out.emplace_back(positions[Index][1]*AngstromsPerNm);
        out.emplace_back(positions[Index][2]*AngstromsPerNm);
        out.emplace_back(charges[Index]*ProtonToCoulomb);
  });
}

inline void getSymbolsByIndex(const std::vector<char>& symbols, const std::vector<int>& indices, std::vector<char>& out)
{
    out.reserve(indices.size()*2);
    for(const auto& Index: indices)
    {
        out.emplace_back(symbols[Index*2]);
        out.emplace_back(symbols[Index*2+1]);
    }
}
inline void getBoxInfo(const std::vector<Vec3>& positions, std::vector<double>& simBoxInfo)
{
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    for (int i=0; i<3; i++)
    {
        for (const auto& pos:positions)
        {
            max = std::max(max, pos[i]);
            min = std::min(min, pos[i]);
        }
        simBoxInfo[i] = max*AngstromsPerNm - min*AngstromsPerNm + 2*AngstromsPerNm;
    }
    simBoxInfo[3] =simBoxInfo[4] = simBoxInfo[5] = 90.0;
}

std::pair<Vec3, Vec3> calculateBoundingBox(const std::vector<Vec3>& positions, const std::vector<int>& Indices, double bbCutoff) {
    Vec3 cutoff ={ bbCutoff, bbCutoff, bbCutoff};
    Vec3 minBounds = {std::numeric_limits<double>::max(),
                                       std::numeric_limits<double>::max(),
                                       std::numeric_limits<double>::max()};
    Vec3 maxBounds = {std::numeric_limits<double>::lowest(),
                                       std::numeric_limits<double>::lowest(),
                                       std::numeric_limits<double>::lowest()};
    
    for (const auto& Index : Indices) {
        for (int i = 0; i < 3; ++i) {
            minBounds[i] = std::min(minBounds[i], positions[Index][i]);
            maxBounds[i] = std::max(maxBounds[i], positions[Index][i]);
        }
    }
    
    return {minBounds-cutoff, maxBounds+cutoff};
}

bool isPointInsideBoundingBox(const Vec3& point,
                              const std::pair<Vec3, Vec3>& boundingBox) {
    const auto& [minBounds, maxBounds] = boundingBox;
    for (int i = 0; i < 3; ++i) {
        if (point[i] < minBounds[i] || point[i] > maxBounds[i]) {
            return false;
        }
    }
    return true;
}


//this function is supposed to filter the relevant MM atoms, 
//which should considerably speed up the calculations on the PuReMD side

inline void filterMMAtomsOMP(const std::vector<Vec3>& positions, const std::vector<int>& mmIndices, const std::pair<Vec3, Vec3>& bbCog, std::vector<int>& relevantIndices)
{
    const int numThreads = omp_get_num_threads();
    std::vector<std::vector<int>> localIndices(numThreads);
    #pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
        std::vector<int>& localVec = localIndices[threadId];
        localVec.reserve(mmIndices.size()/numThreads);
        
        #pragma omp for
        for(size_t i = 0; i < mmIndices.size(); i++)
        {
            const Vec3& point =  positions[mmIndices[i]];
            if(isPointInsideBoundingBox(point, bbCog))
            {  
                localVec.push_back(mmIndices[i]);
            }
        }
    }

    #pragma omp barrier
    size_t totalSize = 0;
    for (const auto& localVec : localIndices) {
        totalSize += localVec.size();
    }
    relevantIndices.reserve(totalSize);
    for (const auto& localVec : localIndices) {
        if (!relevantIndices.empty())
        relevantIndices.insert(relevantIndices.end(), localVec.begin(), localVec.end());
    }    
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
      }
    mmSymbols.emplace_back(symbol[0]);
    mmSymbols.emplace_back(symbol[1]);
    }
}

double ExternalPuremdForceImpl::computeForce(ContextImpl& context, const std::vector<Vec3> &positions, std::vector<Vec3>& forces)
{
  //double factor = context.getReaxffTemperatureRatio();

  // need to seperate positions
  //next we need to seperate and flatten the QM/MM positions and convert to AA#
  int N = owner.getNumAtoms();
  int numQm = qmParticles.size();
  std::vector<double> qmPos, mmPos_q;

  //get the box size. move this into a function
  std::vector<double> simBoxInfo(6);
  getBoxInfo(positions, simBoxInfo);

  //retrieve charges from the context. Had to introduce some changes to classes Context, ContextImpl,
  // UpdateStateDataKernel, CommonUpdateStateDataKernel  
  std::vector<double> charges;
  charges.reserve(N);
  context.getCharges(charges);

  // flatten relevant qm positions and set charges to 0. The last step is important so no exclusions have to be set manually.
  transformPosQM(positions, qmParticles, charges, qmPos);

  //get relevant MM indices from a bounding box sorrounding the ReaxFF atoms
  // 1nm makes total sense as it is the upper taper radius, so interactions will be 0 anyways further away
  double bbCutoff = 1.0;
  std::vector<int> relevantMMIndices;

  auto bbCog =  calculateBoundingBox(positions, qmParticles, bbCutoff);
  // 3nm should be good enough
  filterMMAtomsOMP(positions, mmParticles, bbCog, relevantMMIndices);

  std::vector<char> mmRS;
  int numMm = relevantMMIndices.size();

  getSymbolsByIndex(mmSymbols, relevantMMIndices, mmRS);
  transformPosqMM(positions, charges, relevantMMIndices, mmPos_q);

  context.setCharges(charges);
 
  // OUTPUT VARIABLES
  std::vector<double> qmForces(numQm*3, 0), mmForces(numMm*3,0);
  std::vector<double> qmQ(numQm, 0);

  double energy;

  Interface.getReaxffPuremdForces(numQm, qmSymbols, qmPos,
                                  numMm, mmRS, mmPos_q,
                                  simBoxInfo, qmForces, mmForces, qmQ,
                                  energy);

  // merge the qm and mm forces, additionally transform the scale
  std::vector<Vec3> transformedForces(owner.getNumAtoms(), {0.0, 0.0, 0.0});

  //This is a short operation, no parallelization is needed
  for (size_t i=0; i<qmParticles.size(); ++i)
  {
      transformedForces[qmParticles[i]][0] = qmForces[3*i];
      transformedForces[qmParticles[i]][1] = qmForces[3*i + 1];
      transformedForces[qmParticles[i]][2] = qmForces[3*i + 2];
  }

#pragma omp parallel for
  for (size_t i=0; i<relevantMMIndices.size(); ++i)
  {
      transformedForces[relevantMMIndices[i]][0] = mmForces[i*3];
      transformedForces[relevantMMIndices[i]][1] = mmForces[i*3 + 1];
      transformedForces[relevantMMIndices[i]][2] = mmForces[i*3 + 2];
  }

  //update charges
  //context.setCharges(charges);
  //copy forces and transform from Angstroms * Daltons / ps^2 to kJ/mol/nm

//is around O(n) so parallelization is useful
#pragma omp parallel for
  for(size_t i =0;i<forces.size();++i) {
      forces[i] = -transformedForces[i]*conversionFactor;
  }
  
  //done
  //kCal -> kJ and factor...
  return energy*KJPerKcal;
}
