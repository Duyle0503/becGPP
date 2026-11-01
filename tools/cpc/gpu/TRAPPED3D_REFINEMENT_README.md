# Một cell Kaggle để kiểm tra ca 3D có bẫy — protocol v2

Cập nhật 01/11/2026: đã nhận và kiểm tra dữ liệu thật của protocol này.
Đủ 11/11 lượt vật lý và 109/109 test CUDA đạt trên một Tesla T4. Năng lượng
có bậc hội tụ 1.979/1.987; virial scaled mịn nhất 1.93e-4. Hai khoảng trống
kỹ thuật của bản thảo đã khép, không cần chạy lại để tạo cùng bằng chứng.
Dữ liệu canonical ở data/raw/trapped3d_release_2026_10_31 của bộ CPC; xem
FINAL_READINESS_2026_11_01.md. Các hướng dẫn bên dưới phục vụ tái lập.

Mở `GPU_TRAPPED3D_REFINEMENT.ipynb` trên Kaggle, bật GPU và Internet, rồi chạy
**cell code duy nhất**. Hoặc chép toàn bộ script `.py` cùng tên vào một cell.
Không cần cài becGPP bằng pip: worker tự nhập đúng nguồn được khóa bằng commit
và hash. Không cài lại hoặc thay PyTorch CUDA đang có trong Kaggle.

Đây là protocol mới thay bản sáu job chưa chạy trước đây. Mã tính toán không
thay đổi: becGPP 1.2.2, commit `1c7eef613d147b6857375f9a61342b38cae1bc86`.
27 hash của 19 module và tám tệp test được kiểm tra trước khi chạy. Bản v1 được
giữ trong archive tại máy; không có kết quả GPU nào được tạo giả hoặc gán lại.

## Những gì cell sẽ chạy

Vật lý cố định: 3D, bẫy harmonic `s=2`, `trap_coeff=0.5`, `beta2=100`,
`beta3=0`, `G_C=20`, `Omega=0`, kernel Newton. Đây là vật lý của ca trapped 3D
trong bài, không dùng ca Schrödinger–Newton tự do để thay thế.

| Nhóm | Lượt chạy và cách đối chiếu |
|---|---|
| Regression CUDA | Đủ 109 test release, không skip/loại test; ID khớp bộ CPU CI |
| Lưới | N=96,128,160,192; L=8, padding=2 |
| Hộp | L=6/8/10/12, N=96/128/160/192: cùng dx=0.125; L=8 dùng lại lượt grid N=128 |
| Padding | 2 so với 3 tại N=96 và 128; dùng cùng mảng ban đầu cho mỗi cặp |
| Trạng thái đầu | Gaussian trơn so với TF có noise/rng=0 tại N=128 |
| Sai số cực tiểu hóa | Từ nghiệm N=192, siết residual từ 1e-6 xuống 1e-7 |

Tổng cộng **11 lượt tính vật lý và một lượt regression gồm 109 test**. Các
lượt cùng lưới/hộp được dùng lại, không chạy trùng chỉ để điền bảng.
Seed Gaussian không noise tạo điều kiện đầu xác định. Lượt TF là kiểm tra
nhạy với trạng thái đầu; hai seed không chứng minh cực tiểu toàn cục.

## Điều kiện đạt được khóa trước

- Mọi nghiệm: complex128 trên CUDA, hữu hạn, sai số chuẩn hóa <1e-10, residual
  đạt đúng đích, khối lượng trong hai lớp biên <1e-6 và trong lớp ngoài dày một
  đơn vị chiều dài <1e-8. Tính lại năng lượng, residual và đồng nhất thức mu.
- Hai bậc hội tụ từ các bộ ba lưới không đều phải nằm trong (1.5,2.5). Không
  dùng công thức log2 dành riêng cho lưới giảm bước theo hệ số hai.
- Hai lưới cuối: thay đổi năng lượng và mu scaled <1e-3; RMS radius tương đối
  <5e-3; mật độ đỉnh tương đối <2e-2. Hiệu chỉnh năng lượng của fit h² <1e-3.
- Lưới N=192: virial scaled <5e-4 và ít nhất 12 điểm trên R90.
- Padding: thay đổi năng lượng scaled <2e-11; density L2 tương đối <1e-5;
  hash mảng ban đầu của hai lượt phải giống nhau, kể cả khi resume.
- Hộp: các cặp kế tiếp có thay đổi năng lượng scaled <1e-5 và density L2 trên
  vùng chung <1e-4. So trực tiếp các nút cùng tọa độ, không nội suy/chuẩn hóa lại.
- Hai seed: năng lượng scaled <1e-8 và density L2 <5e-5.
- Lượt siết residual phải đạt 1e-7, thay đổi năng lượng scaled <1e-9. Các gia số
  năng lượng theo lưới phải lớn hơn 100 lần mức sai khác tối ưu hóa đo được
  (có sàn 64 epsilon máy), để tránh diễn giải nhiễu thành bậc hội tụ.

Sai khác năng lượng/mu dùng mẫu số `max(1, abs(reference))`. RMS radius và
mật độ đỉnh dùng sai khác tương đối theo đại lượng tham chiếu; density L2
dùng norm mật độ. R90 được đo quanh tâm bẫy bằng histogram có bin dx/8 và ghi
độ lượng tử hóa này. Quét hộp ở dx=0.125, không phải dx mịn nhất; biên vẫn được
kiểm trên mọi lưới. Các tiêu chí đo độ nhạy hữu hạn lưới cho đúng ca này,
không chứng minh độ chính xác continuum 1e-8 hoặc đúng với mọi tham số.

`overall=PASS` chỉ xuất hiện khi đủ 11 lượt vật lý, toàn bộ so sánh và đủ 109
test CUDA đều đạt. `REVIEW`, `FAIL`, `BLOCKED_MEMORY`, `TIMEOUT`, `INTERRUPTED`
và phần chưa chạy được giữ nguyên. Không hạ lưới hay nới tolerance tự động.

## Tránh chạy lại các phần đã xong

Kết quả ghi vào `/kaggle/working/becgpp_trapped3d_v2_<fingerprint>/`.
Cell kiểm hash của từng tệp trước khi dùng lại kết quả. Chỉ chạy một GPU, mỗi
lượt trong tiến trình riêng để giải phóng bộ nhớ GPU giữa các lượt.

Sau mỗi 256 vòng lặp, worker ghi checkpoint vào hai slot luân phiên và cập
nhật con trỏ bằng thao tác thay tệp. Nếu ngắt giữa lúc ghi, checkpoint đã
commit trước đó vẫn còn. Khi resume, hash điều kiện đầu gốc được giữ riêng
với hash trạng thái bắt đầu attempt mới, để không làm sai đối chiếu padding.
CG/BB được khởi động lại giữa các chặng; thời gian/vòng lặp trong bộ này không
phải phép đo tốc độ của một lượt PCG không ngắt.

Chạy lại cell trong cùng phiên: tự bỏ qua PASS/REVIEW đã hoàn tất và tiếp tục
lượt bị ngắt/timeout/thiếu bộ nhớ. FAIL được thử lại ở attempt mới theo mặc
định; mọi attempt cũ còn nguyên. Không lặp vô hạn trong một lần chạy cell.

Phiên mới: lưu output phiên trước, gắn ZIP kết quả làm Kaggle input, rồi đặt
cấu hình trước phần script/loader:

```python
BECGPP_REFINE_SETTINGS = {
    "resume_zip": "/kaggle/input/TEN_DU_LIEU/TEN_KET_QUA.zip",
    "gpu_index": 0,
    "session_budget_hours": 8.0,
}
```

Nếu output cũ vẫn hiện ngay trong working directory, để `resume_zip=None` và
chạy lại. Không giải nén ZIP cũ đè lên một thư mục có tiến độ mới hơn.
Nếu đổi protocol, phải dùng thư mục khác; cell từ chối trộn kết quả khác nguồn,
khác hash worker/core, khác ngưỡng hay khác cấu hình. Kết quả dùng lại giữ
thông tin GPU/phần mềm thực của phiên đã tạo chúng; không gộp thành benchmark
thời gian trên một GPU giả định.

Đặt `source_dir` nếu dùng repo offline có đúng 27 hash. Mặc định `run_release_tests`
là True. Nếu tắt, refinement vẫn được đánh giá riêng nhưng không có overall
PASS cho việc đóng hai khoảng trống của bài. Ngân sách mặc định của runner là
tám giờ, mỗi job tối đa hai giờ; có thể đặt lại giới hạn thời gian mà không đổi
các ngưỡng khoa học. Đây không phải bảo đảm thời lượng hay quota của Kaggle.

## Lấy kết quả

Sau mỗi job, cell cập nhật ZIP và checksum ở working directory; cuối cell in
`DOWNLOAD` và liên kết. ZIP chứa SUMMARY.json, summary.csv, refinement_table.tex,
convergence.png khi đủ dữ liệu, config, trace, nghiệm, checkpoint, JUnit XML,
log từng attempt, source manifest và môi trường. Không nhúng lại toàn bộ repo.

Lưu phiên/output Kaggle và tải ZIP cùng `.sha256`. Nếu VM bị xóa mà chưa lưu
output, code không thể khôi phục tệp đã mất; ZIP mới nhất giữ các job đã đóng,
còn thư mục working có checkpoint mới hơn của job đang chạy nếu được giữ lại.

Để kiểm tra sau khi nhận ZIP, dùng bộ đọc không cần PyTorch/GPU:

```text
python benchmarks/review_trapped3d_results.py RESULTS.zip --out extracted_results
```

Bộ đọc so hash, receipt, nguồn/code protocol đã biết, JUnit thực và tính lại các
so sánh từ dữ liệu; không chỉ đọc nhãn PASS. Chỉ kết quả GPU thực sau khi chạy
mới có thể dùng để cập nhật bài. Các bài test harness/CPU tại máy không thay
cho kết quả này. `PASS` cũng không đồng nghĩa tạp chí đã chấp nhận bài.

Dùng reader hiện hành đi kèm repo/bộ nộp. Reader cho phép sai khác reduction
cuối bit giữa NumPy 2.1.3/2.4.6 khi tái tạo summary (relative 1e-12, absolute
1e-24), nhưng giữ nguyên mọi ngưỡng khoa học và yêu cầu verdict/boolean trùng
chính xác. Không sửa core/worker đã khóa trong cell thực thi.
