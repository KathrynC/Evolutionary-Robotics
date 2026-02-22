# Replicating the Persona Weight Hypergraph in Mathematica

Instructions for replicating the D3.js visualization (`motion-analytics-toolkit/hero-concepts/08_persona_weight_hypergraph.html`) in Wolfram Mathematica.

## Data File

The source data is:

```
artifacts/persona_hypergraph_curated.json
```

2,315 entries (2,000 fictional characters, 132 celebrities, 79 politicians, 104 mathematicians), 17 k-means clusters, PCA coordinates precomputed.

## 1. Import the Data

```mathematica
raw = Import["~/pybullet_test/Evolutionary-Robotics/artifacts/persona_hypergraph_curated.json", "RawJSON"];

entries = raw["entries"];
clusters = raw["clusters"];
notableNames = raw["notable_names"];

(* 2315 entries, 17 clusters *)
Length[entries]
```

## 2. Entry Fields

Each entry is an Association with these keys:

| Key | Meaning |
|-----|---------|
| `"n"` | Name (e.g., "Harry Potter") |
| `"s"` | Source (e.g., "My Little Pony: Friendship Is Magic") |
| `"c"` | Category: fictional, entertainment, politician, thinker, musician, sports |
| `"w"` | 6D weight vector `{w03, w04, w13, w14, w23, w24}`, each in [-1, 1] |
| `"dx"` | Net X displacement (meters) |
| `"sp"` | Mean speed |
| `"pl"` | Phase lock score [0, 1] |
| `"ef"` | Efficiency (distance per work) |
| `"en"` | Contact entropy (bits) |
| `"px"` | PCA component 1 (precomputed) |
| `"py"` | PCA component 2 (precomputed) |
| `"k"` | Cluster assignment (0-16) |

```mathematica
(* Extract parallel lists *)
names = entries[[All, "n"]];
categories = entries[[All, "c"]];
weights = entries[[All, "w"]];       (* 2315 x 6 matrix *)
pcaCoords = Transpose[{entries[[All, "px"]], entries[[All, "py"]]}];
clusterIds = entries[[All, "k"]];
dxVals = entries[[All, "dx"]];
speeds = entries[[All, "sp"]];
phaseLocks = entries[[All, "pl"]];
entropies = entries[[All, "en"]];
```

## 3. Category Color Map

```mathematica
catColors = <|
  "fictional"     -> RGBColor[0.325, 0.776, 1.0],    (* #53c6ff *)
  "entertainment" -> RGBColor[1.0, 0.635, 0.471],     (* #ffa278 *)
  "politician"    -> RGBColor[1.0, 0.494, 0.651],     (* #ff7ea6 *)
  "thinker"       -> RGBColor[0.827, 0.702, 1.0],     (* #d3b3ff *)
  "musician"      -> RGBColor[0.961, 0.867, 0.471],   (* #f5dd78 *)
  "sports"        -> RGBColor[0.467, 0.941, 0.796]    (* #77f0cb *)
|>;

pointColors = catColors /@ categories;
```

## 4. PCA Scatter Plot (Core Visualization)

```mathematica
scatterPoints = MapThread[
  {PointSize[If[MemberQ[notableNames, #3], 0.005, 0.002]],
   Opacity[If[MemberQ[notableNames, #3], 0.9, 0.5]],
   #1,
   Point[#2]} &,
  {pointColors, pcaCoords, names}
];

Graphics[{
  EdgeForm[None],
  scatterPoints
},
  PlotRange -> All,
  Background -> RGBColor[0.02, 0.04, 0.08],
  ImageSize -> {1200, 800},
  PlotLabel -> Style["Persona Weight Hypergraph — 2,315 in 6D Weight Space",
    White, FontFamily -> "Helvetica", 18],
  Frame -> True,
  FrameStyle -> GrayLevel[0.2],
  FrameLabel -> {
    Style["PC1 (weight space)", GrayLevel[0.4], 11],
    Style["PC2 (weight space)", GrayLevel[0.4], 11]
  }
]
```

## 5. Cluster Convex Hulls

```mathematica
(* Compute convex hull per cluster *)
clusterHulls = Table[
  Module[{pts = Pick[pcaCoords, clusterIds, k]},
    If[Length[pts] >= 3,
      ConvexHullRegion[pts],
      Nothing
    ]
  ],
  {k, 0, 16}
];

(* Cluster colors — HSL wheel *)
clusterColors = Table[
  Hue[Mod[k * 1/17 + 0.04, 1], 0.45, 0.55],
  {k, 0, 16}
];

hullGraphics = MapThread[
  If[#1 =!= Nothing,
    {EdgeForm[{Thin, Opacity[0.3], #2}],
     FaceForm[{Opacity[0.06], #2}],
     MeshPrimitives[#1, 2]},
    {}
  ] &,
  {clusterHulls, clusterColors}
];
```

## 6. Cluster Labels

```mathematica
clusterLabels = Table[
  Module[{
    pts = Pick[pcaCoords, clusterIds, k],
    cl = SelectFirst[clusters, #["id"] == k &]
  },
    If[Length[pts] >= 3,
      Text[
        Style[cl["label"], clusterColors[[k + 1]], FontSize -> 8,
              FontFamily -> "Helvetica"],
        Mean[pts]
      ],
      Nothing
    ]
  ],
  {k, 0, 16}
];
```

## 7. Notable Name Labels

```mathematica
notableEntries = Select[entries, MemberQ[notableNames, #["n"]] &];

notableLabels = Map[
  Text[
    Style[#["n"], catColors[#["c"]], FontSize -> 7, FontFamily -> "Helvetica"],
    {#["px"], #["py"]},
    {0, -1.5}
  ] &,
  notableEntries
];
```

## 8. Combined Plot

```mathematica
Show[
  Graphics[{
    (* Cluster hulls *)
    hullGraphics,
    (* All dots *)
    scatterPoints,
    (* Cluster labels *)
    clusterLabels,
    (* Notable name labels *)
    notableLabels
  },
    PlotRange -> All,
    Background -> RGBColor[0.02, 0.04, 0.08],
    ImageSize -> {1400, 1000},
    PlotLabel -> Style[
      "Persona Weight Hypergraph\n2,315 Humans, Characters & Thinkers in 6D Weight Space",
      White, FontFamily -> "Helvetica", 18, LineSpacing -> {1.5, 0}],
    Frame -> True,
    FrameStyle -> GrayLevel[0.2],
    FrameLabel -> {
      Style["PC1 (weight space)", GrayLevel[0.4], 11],
      Style["PC2 (weight space)", GrayLevel[0.4], 11]
    }
  ]
]
```

## 9. Category Legend

```mathematica
catLegend = SwatchLegend[
  Values[catColors],
  {"Fictional (2,000)", "Entertainment (33)", "Politicians (138)",
   "Thinkers (121)", "Musicians (15)", "Athletes (8)"},
  LegendLabel -> Style["Categories", White, 11],
  LabelStyle -> {GrayLevel[0.7], 9}
];

Legended[%, catLegend]
```

## 10. Bipartite Weight Glyph (for individual entries)

```mathematica
drawWeightGlyph[w_List, size_: 80] := Module[
  {sensorY = {-0.6, 0, 0.6} * size,
   motorY = {-0.3, 0.3} * size,
   sensorX = -0.45 * size, motorX = 0.45 * size,
   wIdx = {{1,2},{3,4},{5,6}}, lines, sensors, motors},

  lines = Flatten@Table[
    Module[{wVal = w[[wIdx[[si, mi]]]], absW},
      absW = Abs[wVal];
      If[absW < 0.02, Nothing,
        {If[wVal > 0, RGBColor[0.467, 0.941, 0.796], RGBColor[1, 0.494, 0.651]],
         Opacity[0.4 + absW * 0.5],
         Thickness[Max[0.001, absW * 0.008]],
         Line[{{sensorX, sensorY[[si]]}, {motorX, motorY[[mi]]}}]}
      ]
    ],
    {si, 1, 3}, {mi, 1, 2}
  ];

  sensors = MapThread[
    {GrayLevel[0.4], Disk[{sensorX, #1}, 3],
     White, Text[Style[#2, 7], {sensorX - 12, #1}]} &,
    {sensorY, {"T", "B", "F"}}
  ];

  motors = MapThread[
    {GrayLevel[0.9], Rectangle[{motorX - 3, #1 - 3}, {motorX + 3, #1 + 3}],
     White, Text[Style[#2, 7], {motorX + 12, #1}]} &,
    {motorY, {"M3", "M4"}}
  ];

  Graphics[{lines, sensors, motors},
    PlotRange -> {{-size, size}, {-size*0.75, size*0.75}},
    ImageSize -> {2*size + 40, 1.5*size + 20},
    Background -> RGBColor[0.03, 0.05, 0.1]
  ]
];

(* Example: draw Trump's weight glyph *)
drawWeightGlyph[{0.8847, -0.5761, 0.9713, -0.4539, 0.7426, -0.6379}]
```

## 11. Recompute PCA from Scratch

If you want to recompute rather than use the precomputed `px`/`py` values:

```mathematica
weightMatrix = N@entries[[All, "w"]];  (* 2315 x 6 *)
centered = weightMatrix - ConstantArray[Mean[weightMatrix], Length[weightMatrix]];
{u, s, v} = SingularValueDecomposition[centered];
pcaFresh = centered . v[[All, {1, 2}]];

(* Verify matches precomputed values *)
ListPlot[pcaFresh, PlotRange -> All, AspectRatio -> 1]
```

## 12. Recompute K-Means from Scratch

```mathematica
clusterResult = FindClusters[
  weightMatrix -> Range[Length[weightMatrix]],
  17,
  Method -> "KMeans",
  DistanceFunction -> EuclideanDistance
];

(* clusterResult is a list of 17 groups of indices *)
```

## 13. Interactive Tooltip Version

```mathematica
interactivePlot = Graphics[
  MapThread[
    {catColors[#3],
     PointSize[If[MemberQ[notableNames, #4], 0.005, 0.002]],
     Opacity[If[MemberQ[notableNames, #4], 0.9, 0.5]],
     Tooltip[
       Point[#1],
       Column[{
         Style[#4, Bold, 14],
         Style[#5, Gray, 10],
         Style["Cluster: " <> ToString[#6], Gray, 9],
         Style["DX: " <> ToString[NumberForm[#7, {5, 2}]] <> "m", 9],
         Style["w: " <> ToString[NumberForm[#, {4, 3}] & /@ #2], 8]
       }]
     ]} &,
    {pcaCoords, weights, categories, names,
     entries[[All, "s"]], clusterIds, dxVals}
  ],
  Background -> RGBColor[0.02, 0.04, 0.08],
  ImageSize -> {1400, 1000},
  PlotRange -> All,
  Frame -> True,
  FrameStyle -> GrayLevel[0.2]
];
```

## 14. Full Interactive Manipulate

```mathematica
catKeys = Keys[catColors];

Manipulate[
  Module[{vis = Select[entries,
    MemberQ[activeCats, #["c"]] &&
    If[notableOnlyQ, MemberQ[notableNames, #["n"]], True] &&
    If[searchStr != "",
       StringContainsQ[ToLowerCase[#["n"]], ToLowerCase[searchStr]],
       True] &]},
    Graphics[
      Map[
        {catColors[#["c"]],
         PointSize[If[MemberQ[notableNames, #["n"]], 0.005, 0.002]],
         Opacity[If[MemberQ[notableNames, #["n"]], 0.9, 0.5]],
         Tooltip[Point[{#["px"], #["py"]}], #["n"]]} &,
        vis
      ],
      PlotRange -> {{-1.8, 1.8}, {-1.9, 1.9}},
      Background -> RGBColor[0.02, 0.04, 0.08],
      ImageSize -> {1000, 700},
      Frame -> True, FrameStyle -> GrayLevel[0.2]
    ]
  ],
  {{activeCats, catKeys, "Categories"}, catKeys, TogglerBar},
  {{notableOnlyQ, False, "Notable Only"}, {True, False}},
  {{searchStr, "", "Search"}, InputField[#, String] &}
]
```

## Feature Comparison: D3 vs Mathematica

| Feature | D3.js (HTML) | Mathematica |
|---------|-------------|-------------|
| Cluster hulls | `d3.polygonHull` | `ConvexHullRegion` + `MeshPrimitives` |
| Hover detail | DOM tooltip + SVG glyph | `Tooltip` (built-in) |
| Search | live text filter | `Manipulate` with `InputField` |
| Category toggle | checkbox DOM manipulation | `Manipulate` with `TogglerBar` |
| Edge modes | SVG lines between notable entries | `Line` primitives with `Manipulate` toggle |
| PCA | Precomputed in Python (numpy) | `SingularValueDecomposition` or `PrincipalComponents` |
| K-means | Precomputed in Python (numpy) | `FindClusters[..., Method -> "KMeans"]` |
