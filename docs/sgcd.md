# SGCD: Photo-Conditioned Causal Stroke-Graph Distillation

## Động cơ

RSED v1 dùng patch evidence trên cùng ink support nên retrieval, attention và random map có cosine rất cao. Student gần như không thay đổi evidence distribution; phần tăng metric nhỏ cũng xuất hiện ở mọi control.

SGCD thay đơn vị giám sát từ patch heatmap sang **đoạn nét có cấu trúc**. Nó dùng photo để đề xuất nét có correspondence, rồi dùng một photo khác để kiểm chứng nét đó có thật sự ảnh hưởng đến teacher retrieval hay không.

Khi `--retrieval_head main`, code giữ đường main: global CLIP descriptor và `3*domain + 1*modality`. SGCD chỉ được kích hoạt bởi `--retrieval_head sgcd`.

## 1. Tạo stroke graph từ raster

Sketch được giảm về lattice 28×28, nhị phân hóa và làm mảnh bằng Zhang–Suen thinning. Skeleton được xem như graph 8 láng giềng:

- endpoint: degree ≤ 1;
- path interior: degree = 2;
- junction: degree > 2.

Các path giữa endpoint/junction được trace riêng. Loop hoặc path quá dài được chia thành các arc. Mỗi sketch giữ tối đa `K=4` path, sau đó map về lattice 7×7 của ViT-B/32.

Đây là graph hình học suy ra từ raster, không phải thứ tự stroke nguyên bản của người vẽ.

## 2. Photo-conditioned local proposal

Với mỗi seen class, main teacher cache chọn hai ảnh gần teacher class centroid nhất. Ảnh đầu dùng cho local proposal; ảnh thứ hai được giữ riêng cho causal verification.

Teacher pool feature cho mỗi sketch path và đo top-patch cosine với dense patches của proposal photo:

```
c_j = mean_top_patch cosine(path_feature_j, photo_patch_feature)
```

Top `P=3` path theo `c_j` trở thành proposal candidates.

## 3. Global causal verification

Mỗi candidate path được mở rộng thành mask priority theo khoảng cách đến path core. Erasure luôn xóa đúng `10%` tổng ink, nên path dài/ngắn không được lợi vì budget.

Với verification photo khác proposal photo:

```
e_j = cosine(T(sketch_clean), T(photo_verify))
    - cosine(T(sketch_without_path_j), T(photo_verify))
```

Verified target là candidate có `e_j` lớn nhất trong top-P local proposals. Các target cache:

- `verified`: local proposal + held-out global causal verification;
- `local`: path tốt nhất theo local correspondence, không verification;
- `random`: structural path khác verified, cùng ink budget;
- `shuffled`: verified target của sketch khác trong minibatch.

Verified confidence bằng normalized positive advantage so với random. Sample không thắng random có confidence 0 và không đóng góp SGCD target loss.

## 4. Teacher-only gate

`kaggle_sgcd_audit.ipy` chạy 512 sketch trước full cache. Mặc định full preparation chỉ được phép khi:

```
verified/random positive-effect ratio >= 1.15
verified beats random rate            >= 0.60
verified/random map cosine            <= 0.75
```

Nếu gate fail, dừng phương pháp ở teacher stage. `--sgcd_force_prepare` chỉ dùng để chẩn đoán, không dùng cho thí nghiệm chính.

## 5. Student

Student đọc dense visual features trước block cuối và dự đoán evidence distribution trên 7×7 ink patches. Evidence descriptor được pool rồi đưa vào retrieval descriptor qua residual khởi tạo bằng 0:

```
z_deploy = normalize(z_main + beta * residual(evidence))
```

Bước 0 có `z_deploy == z_main` chính xác.

Loss:

```
L = L_main + lambda_sgcd * schedule * (
      lambda_where  * Hellinger(student_map, verified_path)
    + lambda_what   * local_similarity_field_alignment
    + lambda_effect * clean_minus_erased_field_alignment
    + lambda_anchor * descriptor_anchor
)
```

Photo student branch được detach trong SGCD. Không có ranking loss hoặc hard-negative mining.

## 6. Diagnostics và tiêu chí

Audit xuất JSON/CSV, effect histogram, path count và montage path/mask. Training xuất:

- deployed/native full unseen mAP và P@100;
- where/what/effect/anchor trên fixed seen batch;
- evidence map cosine và entropy;
- correction norm và deployed/native cosine;
- main/SGCD gradient cosine theo prompt/head;
- teacher/student path heatmaps qua từng epoch.

Contribution chỉ được hỗ trợ nếu:

1. teacher-only gate pass;
2. main+verified > matched main;
3. verified > local/random/shuffled controls;
4. evidence map cosine tăng rõ rệt;
5. kết quả lặp lại ở nhiều student seeds.

## 7. Tham số

- Graph: `sgcd_skeleton_grid`, `sgcd_max_paths`.
- Photo conditioning: `sgcd_photo_representatives`, `sgcd_local_topk_patches`, `sgcd_proposal_topk`.
- Intervention: `sgcd_mask_fraction`.
- Gate: `sgcd_audit_samples`, `sgcd_min_effect_ratio`, `sgcd_min_win_rate`, `sgcd_max_random_map_cosine`.
- Student: `lambda_sgcd_*`, `sgcd_beta`, `sgcd_head_lr`, `sgcd_temperature`.

Đổi graph/photo/mask/teacher/seed/source phải dùng cache mới. Metadata sẽ chặn reuse sai cấu hình.
