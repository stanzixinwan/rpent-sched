from rsched.tegra import parse

THOR = ("11-23-2026 03:43:53 RAM 43776/125809MB (lfb 1054x4MB) SWAP 14/2048MB (cached 2MB) "
        "CPU [3%@972,0%@972,0%@972,5%@972] cpu@37.593C/37.593C gpu@39.843C/39.843C "
        "VDD_GPU 2761mW/2761mW/2761mW VDD_CPU_SOC_MSS 5523mW/5523mW/5523mW "
        "VIN_SYS_5V0 5671mW/5671mW/5671mW VIN 20280mW/10140mW/20280mW")
ORIN = ("RAM 27000/62000MB (lfb 10x4MB) CPU [1%@729,off,2%@729] GR3D_FREQ 37%@[1300] "
        "VDD_GPU_SOC 3200mW/3200mW VDD_CPU_CV 800mW/800mW VIN_SYS_5V0 4000mW/4000mW")


def test_thor_line():
    r = parse(THOR)
    assert r["used_mb"] == 43776 and r["total_mb"] == 125809
    assert r["power_w"] == 20.28 and r["rails"]["VDD_GPU"] == 2.761
    assert r["util_gpu"] is None and r["cpu_util"] == [3.0, 0.0, 0.0, 5.0]


def test_orin_line_without_vin():
    r = parse(ORIN)
    assert r["util_gpu"] == 37.0
    assert abs(r["power_w"] - 4.0) < 1e-9          # VDD rails only
    assert r["cpu_util"] == [1.0, None, 2.0]


def test_not_a_stats_line():
    assert parse("tegrastats: some warning") is None
