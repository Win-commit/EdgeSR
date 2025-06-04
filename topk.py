from experiments import SRExperiment

topk_list = [1, 2, 3, 5, 10]
for topk in topk_list:
    print(f"\n==== Running experiment with topk={topk} ====")
    experiment = SRExperiment(
        sr_model_path="/root/EdgeSR/basemodels/basicSR/hat/HAT_SRx4_ImageNet-pretrain.pth",
        model_type='HAT',
        scale=4
    )
    # 传递 topk 给 run_batch_experiment
    experiment.sr_fusion.detect_important_regions = lambda img_path, **kwargs: \
        experiment.sr_fusion.__class__.detect_important_regions(
            experiment.sr_fusion, img_path, topk=topk
        )
    output_dir = f"/root/autodl-tmp/topk_{topk}"
    experiment.run_batch_experiment(
        lr_dir="/root/autodl-tmp/Datasets/Urban100/image_SRF_4/LR",
        output_dir=output_dir,
        limit=None,
        visualize=False
    )