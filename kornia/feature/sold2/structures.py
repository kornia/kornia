from dataclasses import dataclass, field


@dataclass
class HeatMapRefineCfg:
    mode: str = "local"
    ratio: float = 0.2
    valid_thresh: float = 0.001
    num_blocks: int = 20
    overlap_ratio: float = 0.5


@dataclass
class JunctionRefineCfg:
    num_perturbs: int = 9
    perturb_interval: float = 0.25


@dataclass
class LineDetectorCfg:
    detect_thresh: float = 0.5
    num_samples: int = 64
    inlier_thresh: float = 0.99
    use_candidate_suppression: bool = True
    nms_dist_tolerance: float = 3.0
    heatmap_low_thresh: float = 0.15
    heatmap_high_thresh: float = 0.2
    max_local_patch_radius: float = 3
    lambda_radius: float = 2.0
    use_heatmap_refinement: bool = True
    heatmap_refine_cfg: HeatMapRefineCfg = field(default_factory=HeatMapRefineCfg)
    use_junction_refinement: bool = True
    junction_refine_cfg: JunctionRefineCfg = field(default_factory=JunctionRefineCfg)


@dataclass
class LineMatcherCfg:
    cross_check: bool = True
    num_samples: int = 5
    min_dist_pts: int = 8
    top_k_candidates: int = 10
    grid_size: int = 4
    line_score: bool = False  # True to compute saliency on a line


@dataclass
class BackboneCfg:
    input_channel: int = 1
    depth: int = 4
    num_stacks: int = 2
    num_blocks: int = 1
    num_classes: int = 5


@dataclass
class DetectorCfg:
    backbone_cfg: BackboneCfg = field(default_factory=BackboneCfg)
    use_descriptor: bool = False
    grid_size: int = 8
    keep_border_valid: bool = True
    detection_thresh: float = 0.0153846  # = 1/65: threshold of junction detection
    max_num_junctions: int = 500  # maximum number of junctions per image
    line_detector_cfg: LineDetectorCfg = field(default_factory=LineDetectorCfg)
    line_matcher_cfg: LineMatcherCfg = field(default_factory=LineMatcherCfg)
