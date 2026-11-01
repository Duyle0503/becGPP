# Đánh giá cuối CPC/CPiP — 01/11/2026

**Về kỹ thuật, bản hiện tại đã đủ cơ sở để chuẩn bị nộp CPC/CPiP.** Dữ liệu mới
khép đúng hai điểm còn thiếu: CUDA release 1.2.2 và refinement cho ca Newton
3D có bẫy. Trước khi gửi cần tác giả đọc đúng PDF mới và hoàn tất thủ tục
program files/cổng nộp. Không có bảo đảm được nhận bài.

## Dữ liệu mới và xác minh

Dữ liệu canonical: `data/raw/trapped3d_release_2026_10_31/`.
ZIP gốc: `data/raw/archives/becgpp_trapped3d_v2_5d8306c9b9eb.zip`,
577,163,296 byte, SHA256
`ef61e8fe6d270ceac9b669a526dbfbaedeb7cf0f016e8d78680af6f99aad379a`.
Đã kiểm CRC, 141 hash và tập thành viên ZIP, các tệp giải nén, protocol,
core/worker và 27 hash module/test của release
`1c7eef613d147b6857375f9a61342b38cae1bc86`.
Fingerprint protocol:
`5d8306c9b9eb7cba8bf6458c61aab5aab273191f56372d7176dbf6f1ef2d5b2a`.

**12/12 job PASS = 11 lượt giải vật lý + một job chạy 109 test CUDA.** Đủ
109 ID duy nhất khớp CPU CI, không failure/error/skip. JUnit 28.991 s, khác
worker wall 38.230 s. Một Tesla T4 device 0, Python 3.13.15, PyTorch
2.11.0+cu128, CUDA runtime 12.8, NumPy 2.1.3. Hai T4 được nhìn thấy nhưng
không chạy đa GPU. CUDA 13.0 trong nvidia-smi là khả năng driver.

Reader tính lại mọi tiêu chí, không chỉ đọc nhãn PASS. Khác NumPy 2.1.3/2.4.6
gây ba chênh lệch reduction tối đa 5.294e-23 ở norm mật độ. Đã sửa riêng so
sánh tái tạo summary thành relative 1e-12, absolute 1e-24; mọi ngưỡng khoa học
giữ nguyên, verdict/boolean phải bằng chính xác. 34 kiểm tra protocol/reader
đạt. Không sửa solver, core/worker đã chạy, tám tệp test release hoặc dữ liệu gốc.

## Refinement đúng ca có bẫy

Cố định d=3, c=0.5, s=2, Omega=0, beta2=100, beta3=0, G_C=20, Newton;
float64/complex128. Không dùng ca không bẫy hoặc lưới CPU thô thay thế.

| N, L=8, padding=2 | Năng lượng | Residual tương đối | Virial scaled |
|---|---:|---:|---:|
| 96 | -3.438939917471080 | 6.481e-7 | 7.635e-4 |
| 128 | -3.442149103949376 | 6.457e-7 | 4.322e-4 |
| 160 | -3.443642587809212 | 9.669e-7 | 2.774e-4 |
| 192 | -3.444456016042198 | 8.058e-7 | 1.931e-4 |

Bậc từ các bước lưới không đều là 1.97918 và 1.98710. Chênh năng lượng scaled
giữa hai lưới cuối 2.3616e-4. Fit ba lưới cuối theo h² cho E0=-3.44630075;
hiệu chỉnh scaled tới lưới mịn nhất 5.3528e-4. Đây là ngoại suy hữu hạn, không
phải cận sai số liên tục được chứng minh hoặc độ chính xác liên tục 1e-8.

| Đối chứng | Chênh năng lượng scaled lớn nhất | Chênh mật độ L2 tương đối |
|---|---:|---:|
| Padding 2/3, N=96 và 128, cùng mảng khởi tạo | 1.0322e-15 | 2.5561e-15 |
| Box L=6/8/10/12, cùng h=0.125 | 3.8395e-13 | 3.5958e-7 |
| Gaussian/TF, N=128 | 6.4508e-15 | 2.5734e-7 |
| Residual 1e-6/1e-7, N=192 | 4.3991e-13 | không dùng |

Lượt chặt hơn đạt residual 8.1575e-8. Thay đổi do tối ưu nhỏ hơn bước lưới
cuối hơn tám bậc độ lớn. Box dùng cùng h, crop trùng chính xác, không nội suy.
R90 mới dùng shell histogram tâm bẫy, bin h/8, khác extractor sorted-radius
lịch sử; không trộn các extractor để suy hội tụ.

NumPy/SciPy độc lập, không nhập becGPP, đã tính lại đầy đủ functional và
residual của N=96,128. Chênh năng lượng scaled tối đa 1.292e-15, residual
tuyệt đối tối đa 1.543e-17. Đây là đối chiếu trạng thái rời rạc, không phải
optimizer độc lập mới hoặc chứng minh cực tiểu toàn cục. Mã/JSON trong
`final_revision_2026_11_01/`, entry point portable trong `benchmarks/`.

## Điều chỉnh bản thảo và giới hạn

Bảng refinement và hình mới dùng release 1.2.2. Bỏ bảng hội tụ v1.0 và panel
quét G_C 3D v1.0 khỏi bài hiện hành; dữ liệu gốc vẫn giữ trong lịch sử.
Checkpoint N=128 cũ tính theo functional hiện tại cho E=-3.44214904376, trong
khi CSV cũ ghi -3.44267814867; chênh 5.29105e-4 thuộc thành phần hấp dẫn.
Chưa xác lập nguyên nhân tạo CSV cũ; không quy cho seed, làm tròn hoặc khẳng
định hai phiên bản tương đương. Báo cáo riêng giữ đầy đủ phép tính.

Bài phân biệt 1.1/rc1/release, cập nhật bằng chứng mới thay hai mục pending,
thêm đối chiếu trạng thái độc lập. Supplementary văn bản đã gộp trong bài.
Giữ giới hạn: ca quay 3D chưa có cùng refinement; hai start không chứng minh
cực tiểu toàn cục; mười ca độc lập rc1 chưa đạt đích residual 1e-8; endpoint
quay lịch sử chạm cap vẫn chỉ minh họa. Không suy thành đúng mọi tham số.

Batch gốc vẫn 56 PASS/14 REVIEW/1 FAIL trên 71 job. Đối chứng CPU vẫn 45/48
đạt residual, ba iteration cap. Không đổi nhãn lịch sử, không tuyên bố
default shift/BB luôn nhanh hơn. Thời gian refinement có protocol checkpoint/
restart riêng, không dùng làm benchmark tăng tốc.

## Phù hợp tạp chí và việc còn lại

Bài có Program Summary, MIT license, mô hình, thuật toán, kiểm chứng giải tích,
tham chiếu độc lập, refinement, hiệu năng, ví dụ và dữ liệu phiên bản hóa.
Đóng góp là chương trình cấu hình được và workflow kiểm chứng nhiều chế độ;
không trình bày PCG hay PCG phi địa phương là thuật toán mới.

CPC vẫn đánh giá ý nghĩa, tính mới, ứng dụng vật lý và đóng góp so với phần
mềm có sẵn. Chưa có benchmark đối đầu đầy đủ với các solver xuất bản; đây là
rủi ro biên tập/phản biện, không được giải quyết chỉ bằng 109 test. Phạm vi:
https://shop.elsevier.com/journals/computer-physics-communications/0010-4655 .
Guide for Authors trên ScienceDirect trả 403 ngày 01/11; chưa xác nhận mọi
biểu mẫu hiện hành bằng công cụ này.

Xác nhận tác giả ngày 31/10 thuộc PDF cũ hash 405bfae81f81…; không tự gán cho
PDF có bảng/hình mới. Trạng thái duyệt mới ở `latex/submission/AUTHOR_ACTIONS.md`.
Funding/CRediT/AI/nguyên gốc/không nộp đồng thời đã xác nhận trước đó. Chưa có
xác nhận Program Library deposit/DOI. Chưa gửi bài qua cổng tạp chí.

Kiểm tra PDF và gói có báo cáo máy riêng trong `final_revision_2026_11_01/`.
Integrity PASS và finite-grid PASS không phải quyết định nhận bài.
