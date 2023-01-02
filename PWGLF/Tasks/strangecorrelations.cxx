// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// 2-particle correlations for Xi's task
// =============================
//
// Author: Rik Spijkers (rik.spijkers@cern.ch)
//

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

//use parameters + cov mat non-propagated, aux info + (extension propagated)
using FullTracksExt = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksDCA>;
using FullTracksExtIU = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA>;
using FullTracksExtWithPID = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksDCA, aod::pidTPCPi, aod::pidTPCKa, aod::pidTPCPr>;
using FullTracksExtIUWithPID = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA, aod::pidTPCPi, aod::pidTPCKa, aod::pidTPCPr>;

struct cascadeCorrelations {
  // Basic checks
  HistogramRegistry registry{
    "registry",
    {
      {"hMassXiMinus", "hMassXiMinus", {HistType::kTH1F, {{3000, 0.0f, 3.0f, "Inv. Mass (GeV/c^{2})"}}}},
      {"hMassXiPlus", "hMassXiPlus", {HistType::kTH1F, {{3000, 0.0f, 3.0f, "Inv. Mass (GeV/c^{2})"}}}},
      {"hMassOmegaMinus", "hMassOmegaMinus", {HistType::kTH1F, {{3000, 0.0f, 3.0f, "Inv. Mass (GeV/c^{2})"}}}},
      {"hMassOmegaPlus", "hMassOmegaPlus", {HistType::kTH1F, {{3000, 0.0f, 3.0f, "Inv. Mass (GeV/c^{2})"}}}},

      {"hV0Radius", "hV0Radius", {HistType::kTH1F, {{1000, 0.0f, 100.0f, "cm"}}}},
      {"hCascRadius", "hCascRadius", {HistType::kTH1F, {{1000, 0.0f, 100.0f, "cm"}}}},
      {"hV0CosPA", "hV0CosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hCascCosPA", "hCascCosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hDCAPosToPV", "hDCAPosToPV", {HistType::kTH1F, {{1000, -10.0f, 10.0f, "cm"}}}},
      {"hDCANegToPV", "hDCANegToPV", {HistType::kTH1F, {{1000, -10.0f, 10.0f, "cm"}}}},
      {"hDCABachToPV", "hDCABachToPV", {HistType::kTH1F, {{1000, -10.0f, 10.0f, "cm"}}}},
      {"hDCAV0ToPV", "hDCAV0ToPV", {HistType::kTH1F, {{1000, -10.0f, 10.0f, "cm"}}}},
      {"hDCAV0Dau", "hDCAV0Dau", {HistType::kTH1F, {{1000, 0.0f, 10.0f, "cm^{2}"}}}},
      {"hDCACascDau", "hDCACascDau", {HistType::kTH1F, {{1000, 0.0f, 10.0f, "cm^{2}"}}}},
      {"hLambdaMass", "hLambdaMass", {HistType::kTH1F, {{1000, 0.0f, 10.0f, "Inv. Mass (GeV/c^{2})"}}}},

      {"hPhi", "hPhi", {HistType::kTH1F, {{100, 0, 2*PI, "#varphi"}}}},
      {"hMinPos", "hMinPos", {HistType::kTH1F, {{100, -0.5*PI, 1.5*PI, "#Delta#varphi"}}}},
      {"hMinMin", "hMinMin", {HistType::kTH1F, {{100, -0.5*PI, 1.5*PI, "#Delta#varphi"}}}},
      {"hPosPos", "hPosPos", {HistType::kTH1F, {{100, -0.5*PI, 1.5*PI, "#Delta#varphi"}}}},
    },
  };

  // split into Xi+ and Xi- HAS TO BE HERE BEFORE PROCESS
  Partition<aod::CascDataExt> minCascades = aod::cascdata::sign < 0;
  Partition<aod::CascDataExt> posCascades = aod::cascdata::sign > 0;

  void process(aod::Collision const& collision, aod::CascDataExt const& Cascades, aod::V0sLinked const&, aod::V0Datas const&, FullTracksExtIU const&)
  {
    // partitions are not grouped by default
    auto minCascadesGrouped = minCascades->sliceByCached(aod::cascdata::collisionId, collision.globalIndex());
    auto posCascadesGrouped = posCascades->sliceByCached(aod::cascdata::collisionId, collision.globalIndex());
    
    for (auto& [minCascade, posCascade] : combinations(o2::soa::CombinationsFullIndexPolicy(minCascadesGrouped, posCascadesGrouped))){ // TODO: fix combination policy and grouping

      auto posv0 = posCascade.v0_as<o2::aod::V0sLinked>();
      auto minv0 = minCascade.v0_as<o2::aod::V0sLinked>();

      // not sure why this is needed or why we do this, it seems to cause a segfault
      if (!(posv0.has_v0Data()) || !(minv0.has_v0Data())) {
        return; //skip those cascades for which V0 doesn't exist
      }      
      auto posv0Data = posv0.v0Data();
      auto minv0Data = minv0.v0Data();

      // // Let's try to do some PID
      // // these are the tracks:
      // auto bachTrack = minCascade.bachelor_as<FullTracksExtIUWithPID>();
      // auto posTrack = minv0.posTrack_as<FullTracksExtIUWithPID>();
      // auto negTrack = minv0.negTrack_as<FullTracksExtIUWithPID>();

      // //Bachelor check:
      // if (TMath::Abs(bachTrack.tpcNSigmaPi()) > 3)
      //   continue;
      // //Proton check: 
      // if (TMath::Abs(posTrack.tpcNSigmaPr()) > 3)
      //   continue;
      // //Pion check:
      // if (TMath::Abs(negTrack.tpcNSigmaPi()) > 3)
      //   continue;

      // the ID's so we can make sure we don't double count
      int negBachId = minCascade.bachelorId();
      int negTrackId = minv0.negTrackId();
      int posBachId = posCascade.bachelorId();
      int posTrackId = posv0.posTrackId();

      if(posTrackId != posBachId && negTrackId != negBachId){ // make sure there is no overlap between tracks used
        registry.fill(HIST("hMinPos"), RecoDecay::phi(minCascade.px(), minCascade.py()) - RecoDecay::phi(posCascade.px(), posCascade.py()));
      }
    }

    for (auto& casc : Cascades) {
      if (casc.sign() < 0) { // FIXME: could be done better...
        registry.fill(HIST("hMassXiMinus"), casc.mXi());
        registry.fill(HIST("hMassOmegaMinus"), casc.mOmega());
      } else {
        registry.fill(HIST("hMassXiPlus"), casc.mXi());
        registry.fill(HIST("hMassOmegaPlus"), casc.mOmega());
      }
      // The basic eleven!
      registry.fill(HIST("hV0Radius"), casc.v0radius());
      registry.fill(HIST("hCascRadius"), casc.cascradius());
      registry.fill(HIST("hV0CosPA"), casc.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));
      registry.fill(HIST("hCascCosPA"), casc.casccosPA(collision.posX(), collision.posY(), collision.posZ()));
      registry.fill(HIST("hDCAPosToPV"), casc.dcapostopv());
      registry.fill(HIST("hDCANegToPV"), casc.dcanegtopv());
      registry.fill(HIST("hDCABachToPV"), casc.dcabachtopv());
      registry.fill(HIST("hDCAV0ToPV"), casc.dcav0topv(collision.posX(), collision.posY(), collision.posZ()));
      registry.fill(HIST("hDCAV0Dau"), casc.dcaV0daughters());
      registry.fill(HIST("hDCACascDau"), casc.dcacascdaughters());
      registry.fill(HIST("hLambdaMass"), casc.mLambda());
      registry.fill(HIST("hPhi"), RecoDecay::phi(casc.px(), casc.py()));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<cascadeCorrelations>(cfgc)};
}
