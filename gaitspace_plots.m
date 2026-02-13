(* ::Package:: *)

(* ::Title:: *)
(*Gaitspace Visualizations*)


(* ::Text:: *)
(*Publication-quality figures for Synapse Gait Zoo v2, ported from plot_gaitspace.py.*)
(*Figure 6 is interactive: use pulldown menus to compare any pair of gaits.*)
(*Open this file in Mathematica and evaluate each section sequentially.*)


(* ::Section:: *)
(*Setup & Data Loading*)


(* Robust directory detection: works whether opened as notebook or loaded via Get *)
baseDir = Quiet @ Which[
  StringQ @ NotebookDirectory[], NotebookDirectory[],
  StringQ[$InputFileName], DirectoryName[$InputFileName],
  True, Directory[]];
SetDirectory[baseDir];
Print["Working directory: ", baseDir];
(* Import as RawJSON to get nested Associations instead of Rules *)
raw = Import["synapse_gait_zoo_v2.json", "RawJSON"];
gaits = Flatten @ KeyValueMap[
  Function[{catName, catData},
    KeyValueMap[
      Function[{gn, gd},
        Module[{a = gd["analytics"], p},
          p = If[KeyExistsQ[a, "preserved"], a["preserved"], <||>];
          <|"name" -> gn, "category" -> catName,
            "dx" -> a["outcome"]["dx"],
            "dy" -> a["outcome"]["dy"],
            "yawRad" -> a["outcome"]["yaw_net_rad"],
            "speed" -> a["outcome"]["mean_speed"],
            "speedCV" -> a["outcome"]["speed_cv"],
            "workProxy" -> a["outcome"]["work_proxy"],
            "efficiency" -> a["outcome"]["distance_per_work"],
            "phaseLock" -> a["coordination"]["phase_lock_score"],
            "entropy" -> a["contact"]["contact_entropy_bits"],
            "rollDom" -> a["rotation_axis"]["axis_dominance"][[1]],
            "pitchDom" -> a["rotation_axis"]["axis_dominance"][[2]],
            "yawDom" -> a["rotation_axis"]["axis_dominance"][[3]],
            "axisSwitch" -> a["rotation_axis"]["axis_switching_rate_hz"],
            "attractor" -> Lookup[p, "attractor_type", "unknown"],
            "pareto" -> TrueQ @ Lookup[p, "pareto_optimal", False]|>
        ]],
      catData["gaits"]]],
  raw["categories"]];
Print["Loaded ", Length[gaits], " gaits"]



(* ::Section:: *)
(*Color Scheme & Helpers*)


otherCats = Sort @ Complement[DeleteDuplicates[#["category"] & /@ gaits], {"persona_gaits"}];
allCats = Prepend[otherCats, "persona_gaits"];
catColors = Join[
  <|"persona_gaits" -> GrayLevel[0.5]|>,
  AssociationThread[otherCats -> Table[ColorData[97, i], {i, Length[otherCats]}]]];

notable = <|"43_hidden_cpg_champion" -> "CPG Champion",
  "18_curie" -> "Curie", "44_spinner_champion" -> "Spinner",
  "7_fuller_dymaxion" -> "Fuller", "93_borges_mirror" -> "Borges Mirror",
  "68_grunbaum_penrose" -> "Gr\[UDoubleDot]nbaum", "56_evolved_crab_v2" -> "Crab"|>;

gaitNames = Sort[#["name"] & /@ gaits];
getGait[n_String] := SelectFirst[gaits, #["name"] == n &];

(* Category-colored scatter with notable labels *)
catScatter[xK_String, yK_String, opts___] := Module[{grouped, styles, lbls},
  grouped = Table[
    N @ ({#[xK], #[yK]} & /@ Select[gaits, #["category"] == c &]),
    {c, allCats}];
  styles = Directive[catColors[#], PointSize[Medium], Opacity[0.7]] & /@ allCats;
  lbls = Table[
    If[KeyExistsQ[notable, g["name"]],
      Text[Style[notable[g["name"]], Italic, 8, GrayLevel[0.3]],
        Offset[{10, 8}, {g[xK], g[yK]}]],
      Nothing],
    {g, gaits}];
  ListPlot[grouped, opts, PlotStyle -> styles, Epilog -> lbls,
    Frame -> True, FrameStyle -> GrayLevel[0.3], PlotRange -> All, ImageSize -> 420]];

(* Ternary: a=Roll(top), b=Pitch(bot-left), c=Yaw(bot-right) *)
tXY[{a_, b_, c_}] := With[{s = a + b + c}, {c/s + a/(2 s), a Sqrt[3]/(2 s)}];

(* Export *)
outDir = FileNameJoin[{baseDir, "artifacts", "plots"}];
Quiet @ CreateDirectory[outDir, CreateIntermediateDirectories -> True];
saveFig[fig_, name_String] := Module[{path},
  path = FileNameJoin[{outDir, name}];
  Export[path, fig, ImageResolution -> 200];
  Print["WROTE ", path]];

(* Radar chart builder *)
radarChart[labels_List, datasets_List, colors_List, legends_List] :=
  Module[{n = Length[labels], ang, grid, spokes, aLbl, polys},
    ang = Table[Pi/2 - 2 Pi (k - 1)/n, {k, n}];
    grid = Table[{GrayLevel[0.85], Circle[{0, 0}, r]}, {r, 0.2, 1.0, 0.2}];
    spokes = {GrayLevel[0.85], Line[{{0, 0}, {Cos[#], Sin[#]}}]} & /@ ang;
    aLbl = MapThread[Text[Style[#1, 10], 1.25 {Cos[#2], Sin[#2]}] &, {labels, ang}];
    polys = MapThread[
      Module[{pts, closed},
        pts = MapThread[{#1 Cos[#2], #1 Sin[#2]} &, {#1, ang}];
        closed = Append[pts, First[pts]];
        {{FaceForm[Opacity[0.15, #2]], EdgeForm[{#2, AbsoluteThickness[2]}],
          Polygon[closed]},
         {#2, AbsolutePointSize[8], Point[pts]}}] &,
      {datasets, colors}];
    Legended[
      Graphics[{grid, spokes, aLbl, polys},
        PlotRange -> 1.5 {{-1, 1}, {-1, 1}}, AspectRatio -> 1, ImageSize -> 450],
      Placed[LineLegend[colors, legends, LegendMarkerSize -> 15], Below]]];

(* Category legend (reused in several figures) *)
catLegend = SwatchLegend[
  catColors /@ allCats,
  StringReplace[#, "_" -> " "] & /@ allCats,
  LegendLayout -> "Row", LegendMarkerSize -> {12, 12}];



(* ::Section:: *)
(*Figure 1 \[Dash] Phase Lock Bimodality*)


fig1 = GraphicsRow[{
  Histogram[N[#["phaseLock"] & /@ gaits], {0, 1.05, 0.04},
    Frame -> True, FrameLabel -> {"Phase Lock Score", "Count"},
    PlotLabel -> Style["Phase Lock Distribution (Bimodal)", Bold],
    ChartStyle -> RGBColor[0.298, 0.447, 0.690], ImageSize -> 440],
  catScatter["speed", "phaseLock",
    FrameLabel -> {"Mean Speed", "Phase Lock Score"},
    PlotLabel -> Style["Phase Lock vs Speed", Bold],
    PlotLegends -> None, ImageSize -> 440]
}, ImageSize -> 920];
saveFig[fig1, "fig01_phase_lock_bimodal.png"];
fig1



(* ::Section:: *)
(*Figure 2 \[Dash] Contact Entropy Independence*)


fig2 = Module[{pairs, panels},
  pairs = {{"speed", "Speed"}, {"phaseLock", "Phase Lock"}, {"efficiency", "Efficiency"}};
  panels = Table[
    Module[{xs, ys, r, extra = {}},
      xs = N[#[p[[1]]] & /@ gaits];
      ys = N[#["entropy"] & /@ gaits];
      If[p[[1]] === "efficiency",
        Module[{clip, ok},
          clip = Quantile[xs, 0.97];
          ok = Select[Transpose[{xs, ys}], #[[1]] <= clip &];
          r = Correlation[ok[[All, 1]], ok[[All, 2]]];
          extra = {PlotRange -> {{-0.05 clip, 1.15 clip}, All}};],
        r = Correlation[xs, ys]];
      catScatter[p[[1]], "entropy",
        FrameLabel -> {p[[2]], "Contact Entropy (bits)"},
        PlotLabel -> Style[Row[{"r = ", NumberForm[r, {4, 3}]}], Bold],
        PlotLegends -> None, ImageSize -> 350,
        Sequence @@ extra]],
    {p, pairs}];
  Column[{Style["Contact Entropy Is Independent of Performance", Bold, 13],
    GraphicsRow[panels, ImageSize -> 1100]}, Alignment -> Center]];
saveFig[fig2, "fig02_contact_entropy_independence.png"];
fig2



(* ::Section:: *)
(*Figure 3 \[Dash] Axis Dominance*)


fig3 = Module[{tri, ternPlot, counts, barPlot},
  tri = tXY /@ {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
  ternPlot = Graphics[{
    {Black, Line[tri]},
    Text[Style["Roll", Bold, 11], tri[[1]] + {0, 0.04}, {0, -1}],
    Text[Style["Pitch", Bold, 11], tri[[2]] + {-0.03, -0.03}, {1, 1}],
    Text[Style["Yaw", Bold, 11], tri[[3]] + {0.03, -0.03}, {-1, 1}],
    (* Gait points by category *)
    Table[Module[{pts = Select[gaits, #["category"] == c &]},
      {catColors[c], PointSize[Medium], Opacity[0.7],
       Point[tXY[{#["rollDom"], #["pitchDom"], #["yawDom"]}] & /@ pts]}],
      {c, allCats}],
    (* Notable labels *)
    Table[If[KeyExistsQ[notable, g["name"]],
      Text[Style[notable[g["name"]], Italic, 7, GrayLevel[0.3]],
        Offset[{8, 4}, tXY[{g["rollDom"], g["pitchDom"], g["yawDom"]}]]],
      Nothing], {g, gaits}]},
    AspectRatio -> Automatic, PlotLabel -> Style["Axis Dominance Simplex", Bold],
    ImageSize -> 420];

  counts = {
    Length @ Select[gaits, #["rollDom"] >= #["pitchDom"] && #["rollDom"] >= #["yawDom"] &],
    Length @ Select[gaits, #["pitchDom"] > #["rollDom"] && #["pitchDom"] >= #["yawDom"] &],
    Length @ Select[gaits, #["yawDom"] > #["rollDom"] && #["yawDom"] > #["pitchDom"] &]};
  barPlot = BarChart[counts,
    ChartLabels -> Placed[{"Roll", "Pitch", "Yaw"}, Axis],
    ChartStyle -> {RGBColor[0.91, 0.30, 0.24], RGBColor[0.20, 0.60, 0.86], RGBColor[0.18, 0.80, 0.44]},
    LabelingFunction -> Above, Frame -> True,
    FrameLabel -> {None, "Number of Gaits"},
    PlotLabel -> Style["Dominant Rotation Axis", Bold],
    ImageSize -> 380];
  GraphicsRow[{ternPlot, barPlot}, ImageSize -> 850]];
saveFig[fig3, "fig03_axis_dominance.png"];
fig3



(* ::Section:: *)
(*Figure 4 \[Dash] Speed vs Efficiency*)


fig4 = Module[{sp, ef, q1, q3, fence, ylim, medS, medE, main, mLines, stars},
  sp = N[#["speed"] & /@ gaits];
  ef = N[#["efficiency"] & /@ gaits];
  {q1, q3} = Quantile[ef, {1/4, 3/4}];
  fence = q3 + 5 (q3 - q1);
  ylim = Max[Select[ef, # <= fence &]] * 1.25;
  medS = Median[sp]; medE = Median[ef];

  main = catScatter["speed", "efficiency",
    FrameLabel -> {"Mean Speed", "Distance per Work (Efficiency)"},
    PlotLabel -> Style["Speed\[Dash]Efficiency Landscape", Bold],
    PlotRange -> {{Automatic, Automatic}, {-0.03 ylim, ylim}},
    PlotLegends -> None, ImageSize -> 600];

  mLines = Graphics[{GrayLevel[0.7], Dashed, AbsoluteThickness[0.8],
    Line[{{medS, -ylim}, {medS, ylim}}],
    Line[{{-1, medE}, {5, medE}}]}];

  stars = ListPlot[
    {#["speed"], #["efficiency"]} & /@ Select[gaits, #["pareto"] &],
    PlotMarkers -> {"\[FivePointedStar]", 14},
    PlotStyle -> Directive[RGBColor[1, 0.84, 0], EdgeForm[Black]]];

  Show[main, mLines, stars,
    PlotRange -> {{Automatic, Automatic}, {-0.03 ylim, ylim}},
    Epilog -> Table[
      If[KeyExistsQ[notable, g["name"]] && g["efficiency"] <= ylim,
        Text[Style[notable[g["name"]], Italic, 8, GrayLevel[0.3]],
          Offset[{10, 8}, {g["speed"], g["efficiency"]}]],
        Nothing], {g, gaits}]]];
saveFig[fig4, "fig04_speed_efficiency.png"];
fig4



(* ::Section:: *)
(*Figure 5 \[Dash] Champion Comparison*)


fig5 = Module[{trio, metrics, ranges, norm, data},
  trio = {"CPG Champion" -> getGait["43_hidden_cpg_champion"],
    "Curie" -> getGait["18_curie"],
    "Spinner" -> getGait["44_spinner_champion"]};
  metrics = {"speed" -> "Speed", "efficiency" -> "Efficiency",
    "phaseLock" -> "Phase Lock", "entropy" -> "Entropy",
    "rollDom" -> "Roll Dom.", "axisSwitch" -> "Axis Switch"};
  ranges = Association @ Table[
    Module[{v = N[#[k[[1]]] & /@ gaits]},
      k[[1]] -> {Quantile[v, 0.02], Quantile[v, 0.98]}],
    {k, metrics}];
  norm[val_, key_] := Clip[(val - ranges[key][[1]]) / (ranges[key][[2]] - ranges[key][[1]]), {0, 1}];

  (* Each row = one metric group, columns = champions *)
  data = Table[
    Table[norm[t[[2]][m[[1]]], m[[1]]], {t, trio}],
    {m, metrics}];

  BarChart[data, ChartLayout -> "Grouped",
    ChartLabels -> {metrics[[All, 2]], None},
    ChartLegends -> trio[[All, 1]],
    ChartStyle -> {RGBColor[0.91, 0.30, 0.24], RGBColor[0.20, 0.60, 0.86],
      RGBColor[0.18, 0.80, 0.44]},
    Frame -> True, FrameLabel -> {None, "Normalized Score"},
    PlotLabel -> Style["Champion Comparison: CPG vs Curie vs Spinner", Bold],
    PlotRange -> {Automatic, {0, 1.15}},
    ImageSize -> 750]];
saveFig[fig5, "fig05_champion_comparison.png"];
fig5



(* ::Section:: *)
(*Figure 6 \[Dash] Topology Bifurcation (Static Export)*)


fig6static = Module[{g1, g2, labels, v1, v2, mx, red, blue},
  g1 = getGait["43_hidden_cpg_champion"];
  g2 = getGait["44_spinner_champion"];
  labels = {"Speed", "|DX|", "Phase Lock", "Roll Dom.", "Entropy", "|Yaw|"};
  v1 = {g1["speed"], Abs[g1["dx"]], g1["phaseLock"], g1["rollDom"], g1["entropy"], Abs[g1["yawRad"]]};
  v2 = {g2["speed"], Abs[g2["dx"]], g2["phaseLock"], g2["rollDom"], g2["entropy"], Abs[g2["yawRad"]]};
  mx = MapThread[Max[#1, #2, 0.001] &, {v1, v2}];
  red = RGBColor[0.91, 0.30, 0.24]; blue = RGBColor[0.20, 0.60, 0.86];
  Column[{
    Style["Same Topology, Opposite Behavior \[Dash] Gait 43 vs 44", Bold, 13],
    radarChart[labels, {v1/mx, v2/mx}, {red, blue},
      {"Gait 43 (CPG Champion)", "Gait 44 (Spinner)"}]
  }, Alignment -> Center]];
saveFig[fig6static, "fig06_topology_bifurcation.png"];
fig6static



(* ::Section:: *)
(*Figure 6 \[Dash] Interactive Gait Comparison*)


(* ::Text:: *)
(*Use the pulldown menus below to compare any two gaits on 6 normalized axes.*)


Manipulate[
  Module[{g1, g2, labels, v1, v2, mx, red, blue},
    g1 = getGait[name1]; g2 = getGait[name2];
    labels = {"Speed", "|DX|", "Phase Lock", "Roll Dom.", "Entropy", "|Yaw|"};
    v1 = {g1["speed"], Abs[g1["dx"]], g1["phaseLock"],
      g1["rollDom"], g1["entropy"], Abs[g1["yawRad"]]};
    v2 = {g2["speed"], Abs[g2["dx"]], g2["phaseLock"],
      g2["rollDom"], g2["entropy"], Abs[g2["yawRad"]]};
    mx = MapThread[Max[#1, #2, 0.001] &, {v1, v2}];
    red = RGBColor[0.91, 0.30, 0.24]; blue = RGBColor[0.20, 0.60, 0.86];
    Column[{
      Style[Row[{name1, "  vs  ", name2}], Bold, 14],
      radarChart[labels, {v1/mx, v2/mx}, {red, blue}, {name1, name2}],
      (* Raw values table *)
      Grid[
        Prepend[
          Transpose[{labels, NumberForm[#, {5, 3}] & /@ v1, NumberForm[#, {5, 3}] & /@ v2}],
          {Style["Metric", Bold], Style[name1, Bold, red], Style[name2, Bold, blue]}],
        Frame -> All, FrameStyle -> GrayLevel[0.8],
        Background -> {None, {GrayLevel[0.95], {White}}},
        Spacings -> {2, 0.8}]
    }, Alignment -> Center, Spacings -> 1]],
  {{name1, "43_hidden_cpg_champion", "Gait 1"}, gaitNames, PopupMenu},
  {{name2, "44_spinner_champion", "Gait 2"}, gaitNames, PopupMenu},
  ControlPlacement -> Top]



(* ::Section:: *)
(*Figure 7 \[Dash] Category Overview*)


fig7 = Module[{projs, grid},
  projs = {
    {"speed", "phaseLock", "Speed", "Phase Lock"},
    {"speed", "entropy", "Speed", "Contact Entropy"},
    {"rollDom", "pitchDom", "Roll Dominance", "Pitch Dominance"},
    {"dx", "yawRad", "DX (displacement)", "Yaw (rad)"}};
  grid = GraphicsGrid[
    Partition[
      Table[catScatter[p[[1]], p[[2]],
        FrameLabel -> {p[[3]], p[[4]]},
        PlotLegends -> None, ImageSize -> 400],
        {p, projs}], 2],
    ImageSize -> 860, Spacings -> 5];
  Column[{
    Style["Gaitspace Projections by Category", Bold, 13],
    grid, catLegend}, Alignment -> Center, Spacings -> 1]];
saveFig[fig7, "fig07_category_overview.png"];
fig7
