import json, os, glob

json_dir = 'Siam-NestedUNet/tmp/json_files'
files = sorted(glob.glob(os.path.join(json_dir, 'metadata_epoch_*.json')))

results = []
for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    vm = data.get('validation_metrics', {})
    epoch = os.path.basename(f).replace('metadata_epoch_', '').replace('.json', '')
    results.append({
        'epoch': int(epoch),
        'file': os.path.basename(f),
        'loss': vm.get('cd_losses', float('inf')),
        'f1': vm.get('cd_f1scores', 0),
        'precision': vm.get('cd_precisions', 0),
        'recall': vm.get('cd_recalls', 0),
        'corrects': vm.get('cd_corrects', 0),
        'lr': vm.get('learning_rate', 0),
    })

print('=' * 80)
print(f'共分析 {len(results)} 个 epoch 的模型数据')
print('=' * 80)

# 按 F1 分数排序
results_sorted_f1 = sorted(results, key=lambda x: x['f1'], reverse=True)
print('\n【按 F1 Score 排名 Top 10】')
print(f'{"排名":<5} {"Epoch":<8} {"F1 Score":<12} {"Precision":<12} {"Recall":<12} {"Loss":<12} {"Corrects":<12}')
print('-' * 73)
for i, r in enumerate(results_sorted_f1[:10]):
    print(f'{i+1:<5} {r["epoch"]:<8} {r["f1"]:<12.6f} {r["precision"]:<12.6f} {r["recall"]:<12.6f} {r["loss"]:<12.6f} {r["corrects"]:<12.4f}')

# 按 Loss 排序（越低越好）
results_sorted_loss = sorted(results, key=lambda x: x['loss'])
print('\n【按 Loss 排名 Top 10 (越低越好)】')
print(f'{"排名":<5} {"Epoch":<8} {"Loss":<12} {"F1 Score":<12} {"Precision":<12} {"Recall":<12} {"Corrects":<12}')
print('-' * 73)
for i, r in enumerate(results_sorted_loss[:10]):
    print(f'{i+1:<5} {r["epoch"]:<8} {r["loss"]:<12.6f} {r["f1"]:<12.6f} {r["precision"]:<12.6f} {r["recall"]:<12.6f} {r["corrects"]:<12.4f}')

# 按 Precision 排序
results_sorted_pre = sorted(results, key=lambda x: x['precision'], reverse=True)
print('\n【按 Precision 排名 Top 10】')
print(f'{"排名":<5} {"Epoch":<8} {"Precision":<12} {"F1 Score":<12} {"Recall":<12} {"Loss":<12}')
print('-' * 61)
for i, r in enumerate(results_sorted_pre[:10]):
    print(f'{i+1:<5} {r["epoch"]:<8} {r["precision"]:<12.6f} {r["f1"]:<12.6f} {r["recall"]:<12.6f} {r["loss"]:<12.6f}')

# 按 Recall 排序
results_sorted_rec = sorted(results, key=lambda x: x['recall'], reverse=True)
print('\n【按 Recall 排名 Top 10】')
print(f'{"排名":<5} {"Epoch":<8} {"Recall":<12} {"F1 Score":<12} {"Precision":<12} {"Loss":<12}')
print('-' * 61)
for i, r in enumerate(results_sorted_rec[:10]):
    print(f'{i+1:<5} {r["epoch"]:<8} {r["recall"]:<12.6f} {r["f1"]:<12.6f} {r["precision"]:<12.6f} {r["loss"]:<12.6f}')

# 综合最优模型 (F1 最高)
best = results_sorted_f1[0]
print('\n' + '=' * 80)
print('【综合最优模型 (以 F1 Score 为主要指标)】')
print('=' * 80)
print(f'  Epoch:         {best["epoch"]}')
print(f'  F1 Score:      {best["f1"]:.6f}')
print(f'  Precision:     {best["precision"]:.6f}')
print(f'  Recall:        {best["recall"]:.6f}')
print(f'  Loss:          {best["loss"]:.6f}')
print(f'  Corrects:      {best["corrects"]:.4f}%')
print(f'  Learning Rate: {best["lr"]}')
print(f'  文件名:        {best["file"]}')
print('=' * 80)
