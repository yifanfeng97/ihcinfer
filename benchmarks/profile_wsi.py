import cProfile, pstats, time
from ihcinfer import IHCAnalyzer

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
SVS = "tests/data/slides/98140-6 CD3.svs"

def main():
    inf = IHCAnalyzer(model_dir=MODEL_DIR, gpu_ids=[3], batch_size=8)
    t0 = time.perf_counter()
    result = inf.infer_wsi(
        SVS,
        "outputs/wsi_profile_out",
        num_region_samples=0,
        num_patch_samples=0,
        skip_thumbnail=True,
    )
    print(f"WSI core time: {time.perf_counter()-t0:.2f}s, records: {len(result.records)}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("outputs/wsi_profile.stats")
