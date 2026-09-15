# Retrieval-Conditioned Stroke Evidence Distillation (RSED)

## Mục tiêu

RSED truyền từ teacher sang student phần bằng chứng cục bộ trên nét sketch có ảnh hưởng trực tiếp đến trường tương đồng sketch–photo. Phương pháp không xem attention map là đích cuối. Attention chỉ là một prior để ổn định attribution; descriptor dùng để retrieval nhận một residual được pool từ chính evidence map mà student dự đoán.

Khi `--retrieval_head main`, đường forward, loss domain/modality và descriptor giữ hành vi main. Khi `--retrieval_head rsed`, nhánh RSED chỉ thay đổi sketch descriptor; photo descriptor vẫn là main.

## Teacher target

Với sketch `s`, teacher sketch descriptor `z_T(s)` và prototype ảnh của lớp seen `p_T(y)`, score mục tiêu là

```
r_T(s,y) = cosine(z_T(s), p_T(y)).
```

Ở residual stream trước block thị giác cuối, mỗi patch `i` nhận attribution

```
a_i = positive(<h_i, d r_T / d h_i>)
```

với trị tuyệt đối làm fallback khi toàn bộ contribution có dấu âm. Target retrieval evidence kết hợp attribution, final-block CLS attention và soft ink support:

```
q_i ∝ a_i * sqrt(attention_i) * sqrt(ink_i).
```

Map được resize về lattice 7×7 của ViT-B/32 và khuếch tán qua graph bốn láng giềng chỉ khi hai patch nằm trên ink support. Vì Sketchy chỉ cung cấp raster, graph này là xấp xỉ topology nét; nó không giả vờ khôi phục thứ tự stroke.

Hai control có cùng support và budget được cache cùng lúc:

- `attention`: raw CLS attention, không retrieval-conditioned gradient.
- `random`: random score trên ink support.
- `shuffled`: target retrieval của một sketch khác trong batch.

Teacher còn cache local evidence descriptor và descriptor sau khi xóa đúng một tỷ lệ ink theo từng target.

## Student descriptor

Student đọc patch residuals trước visual block cuối. Một query từ native CLS và keys từ patch features tạo evidence distribution `w_S`. Evidence descriptor là weighted pooling của patch features. Descriptor triển khai là

```
z_S^deploy = normalize(z_S^main + beta * residual(e_S)).
```

Layer cuối của residual được khởi tạo bằng 0. Vì vậy tại bước 0, `z_S^deploy` bằng chính xác `z_S^main`; nhánh mới chỉ can thiệp khi học được tín hiệu.

## Loss

```
L = L_main + lambda_rsed * schedule * L_RSED

L_RSED = lambda_where  * L_where
       + lambda_what   * L_what
       + lambda_effect * L_effect
       + lambda_anchor * L_anchor
```

- `L_where`: Hellinger giữa evidence distribution student và teacher.
- `L_what`: căn chỉnh trường tương đồng centered giữa student local-evidence descriptor với các photo student, và teacher local-evidence descriptor với các photo teacher.
- `L_effect`: xóa đúng budget ink do teacher chọn; căn chỉnh hướng thay đổi của toàn bộ trường tương đồng clean-minus-masked. Photo branch của student được detach để tránh nghiệm photo tự di chuyển theo loss này.
- `L_anchor`: giữ deployed sketch descriptor gần native main descriptor để hạn chế phá hủy geometry CLIP ban đầu.

Không có ranking loss, hard-negative mining hay nhãn unseen trong RSED target.

## Diagnostics

`--rsed_diagnostics` ghi vào `tb_logs/<run>/version_*/rsed_diagnostics`:

- full unseen deployed và native mAP/P@100;
- fixed seen-batch component losses;
- map, local-field và counterfactual-effect cosine;
- evidence entropy, correction norm, deployed/native cosine;
- cosine và norm ratio giữa gradient main và RSED trên sketch prompts, photo prompts và evidence head;
- teacher/student heatmaps, difference maps và teacher-selected erasures;
- teacher target confidence, attention/random overlap, entropy và exact removed-ink fraction.

Một kết quả chỉ hỗ trợ contribution khi retrieval target vượt matched main và vượt attention/random/shuffled controls. Giảm component loss một mình không đủ.

## Tham số chính

- `--lambda_rsed`: trọng số tổng.
- `--lambda_rsed_where`, `--lambda_rsed_what`, `--lambda_rsed_effect`, `--lambda_rsed_anchor`.
- `--rsed_beta`: biên độ residual đi vào retrieval descriptor.
- `--rsed_head_lr`: learning rate riêng của evidence head.
- `--rsed_temperature`: độ sắc của student evidence map.
- `--rsed_mask_fraction`: tỷ lệ ink bị xóa cho causal effect.
- `--rsed_graph_steps`, `--rsed_graph_mix`: mức khuếch tán theo graph nét.
- `--rsed_warmup_epochs`, `--rsed_decay_start_epoch`: schedule.
- `--rsed_target`: retrieval/attention/random/shuffled.

Cache phải đổi tên hoặc rebuild khi thay mask fraction, graph, ink threshold/softness, teacher config, seed hoặc source.
