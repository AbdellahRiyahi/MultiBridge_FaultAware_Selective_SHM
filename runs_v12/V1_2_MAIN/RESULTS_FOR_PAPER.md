# MultiBridge V1.2 — COMPLETE_WITH_LIMITATIONS
Étude : `V1_2_MAIN`
Mode : `full`
Résultats principaux calculés : **True**
Toutes les expériences demandées terminées : **True**

Analyse révisée sur un benchmark déjà examiné en V1.0 : aucune nouvelle confirmation indépendante. Les intervalles sont conditionnels aux campagnes observées.

## Résumé calculé
```csv
task,model,subset,n_acquisitions,coverage,coverage_reference,coverage_altered,selective_risk,false_structural_alarm_rate,false_no_alert_rate,auroc_all
Z24,FaultAware,nominal,144,0.7847222222222222,0.2222222222222222,0.8650793650793651,0.0,0.0,0.0,0.9872134038800706
Z24,FaultAware,seen_faults,144,0.7699074074074074,0.2962962962962963,0.8375661375661375,0.0018039687312086,0.0111111111111111,0.0,0.9780188124632568
Z24,FaultAware,unseen_fault,144,0.7708333333333334,0.2777777777777778,0.8412698412698413,0.006006006006006,0.037037037037037,0.0,0.9738389182833628
Z24,FaultAware,all_faults,144,0.7700617283950617,0.2932098765432099,0.8381834215167548,0.00250501002004,0.015432098765432,0.0,0.9771594051429444
Z24,StandardNN,nominal,144,0.7916666666666666,0.2777777777777778,0.8650793650793651,0.0,0.0,0.0,0.984567901234568
Z24,StandardNN,seen_faults,144,0.7685185185185185,0.3296296296296296,0.8312169312169312,0.0012048192771084,0.0074074074074074,0.0,0.9752694493435234
Z24,StandardNN,unseen_fault,144,0.7638888888888888,0.4259259259259259,0.8121693121693122,0.0,0.0,0.0,0.9765824025083284
Z24,StandardNN,all_faults,144,0.7677469135802469,0.345679012345679,0.828042328042328,0.0010050251256281,0.0061728395061728,0.0,0.9751984126984128
Route345,FaultAware,nominal,25,0.64,0.6666666666666666,0.631578947368421,0.125,0.3333333333333333,0.0,0.7719298245614036
Route345,FaultAware,seen_faults,25,0.6506666666666666,0.6666666666666666,0.6456140350877193,0.1229508196721311,0.3333333333333333,0.0,0.7716179337231969
Route345,FaultAware,unseen_fault,25,0.6133333333333333,0.5,0.6491228070175439,0.0652173913043478,0.1666666666666666,0.0,0.7865497076023391
Route345,FaultAware,all_faults,25,0.6444444444444445,0.6388888888888888,0.6461988304093568,0.1137931034482758,0.3055555555555556,0.0,0.7733376651505306
Route345,StandardNN,nominal,25,0.6,0.5,0.631578947368421,0.0666666666666666,0.1666666666666666,0.0,0.780701754385965
Route345,StandardNN,seen_faults,25,0.6,0.5,0.631578947368421,0.0666666666666666,0.1666666666666666,0.0,0.7781676413255361
Route345,StandardNN,unseen_fault,25,0.6,0.5,0.631578947368421,0.0666666666666666,0.1666666666666666,0.0,0.787524366471735
Route345,StandardNN,all_faults,25,0.6,0.5,0.631578947368421,0.0666666666666666,0.1666666666666666,0.0,0.7789148797920729
openLAB2025,FaultAware,nominal,12,0.6666666666666666,0.5,0.8333333333333334,0.0,0.0,0.0,0.9444444444444444
openLAB2025,FaultAware,seen_faults,12,0.5611111111111111,0.4555555555555555,0.6666666666666666,0.0099009900990099,0.0111111111111111,0.0,0.9638271604938272
openLAB2025,FaultAware,unseen_fault,12,0.6944444444444444,0.5555555555555556,0.8333333333333334,0.08,0.1111111111111111,0.0,0.9444444444444444
openLAB2025,FaultAware,all_faults,12,0.5833333333333334,0.4722222222222222,0.6944444444444444,0.0238095238095238,0.0277777777777777,0.0,0.9604766803840878
openLAB2025,StandardNN,nominal,12,0.75,0.6666666666666666,0.8333333333333334,0.1111111111111111,0.1666666666666666,0.0,0.9444444444444444
openLAB2025,StandardNN,seen_faults,12,0.5166666666666667,0.4555555555555555,0.5777777777777777,0.043010752688172,0.0444444444444444,0.0,0.9475308641975309
openLAB2025,StandardNN,unseen_fault,12,0.6388888888888888,0.4444444444444444,0.8333333333333334,0.0,0.0,0.0,0.9444444444444444
openLAB2025,StandardNN,all_faults,12,0.5370370370370371,0.4537037037037037,0.6203703703703703,0.0344827586206896,0.037037037037037,0.0,0.9468449931412894

```

## Limites
- Endpoint: documented structural change versus reference, not RUL, optimal maintenance or safety certification.
- Nominal recordings may have unannotated sensor problems; truth for quality evaluation means added synthetic corruption.
- Three outer folds test every retained acquisition once. Model training sets overlap between folds; neither folds nor windows are independent new campaigns.
- Route345 and openLAB have few independent reference acquisitions. Low-N calibration uses identity probabilities and a fixed 0.5 structural threshold; abstention calibration remains empirical.
- Group/state-cluster bootstrap intervals are conditional descriptive intervals from frozen OOF predictions, not a population guarantee.
- The primary evaluation does not withhold entire damage states or entire new bridges from the structural model.
- openLAB 2025 uses only the two verified velocity response channels, random excitation, A40 and the accepted hardware regime; other recordings remain explicitly outside scope.
- Losing one of two openLAB response sensors may make a structural decision unavailable; rejection is not a correct diagnosis.
- Corruptions are local to source windows after source-native preprocessing; they do not reconstruct a persistent hardware failure across the complete recording.
- A14 exclusion is symmetric and based on previously reported project QC; no new independent physical-fault label is claimed.
- openLAB 2024 is a separate event-triggered reference experiment; alarm fractions are not continuous hourly rates.
- Cross-bridge quality transfer uses shared synthetic corruption templates. Good results do not establish field fault transfer.
- V1.2 revises a benchmark whose V1.0 test results have already been inspected. This is not a new independent confirmatory test.
- Sensor-check thresholds are acquisition-level empirical calibrations; small calibration cohorts do not establish a population false-alarm guarantee.
- Confidence scores use genuine ties; connecting points in ranking plots does not create attainable intermediate coverages.
- Quality cross-fitting is the primary separately labelled training experiment, not an automatic improvement claim.
- The same architecture/configuration is used on CPU and CUDA, but floating-point results need not be bit-for-bit identical.
- All performance values come from this execution. Statistical superiority, causal damage identification and conference acceptance are not guaranteed.