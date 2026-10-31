# Đánh giá bản sửa cuối CPC/CPiP — 31/10/2026

**Kết luận: bài đã được sửa sâu và có bộ hồ sơ kiểm tra cụ thể, nhưng tôi chưa
khuyên bấm nộp ngay.** Điểm còn yếu quan trọng là thiếu CUDA đúng release 1.2.2,
chưa có refinement đủ cho đúng ca 3D có bẫy. Cả hai tác giả đã được người dùng
xác nhận duyệt bản PDF cuối ngày 31/10. Không thể bảo đảm được nhận hoặc “không thể bắt bẻ”.

## Đã thay đổi

- Nêu đóng góp là phần mềm tích hợp có kiểm chứng. Không tuyên bố PCG hay PCG
  với tương tác phi địa phương là thuật toán mới, hoặc hơn mọi phần mềm khác.
- Sửa diễn giải critical rotation: tổ hợp kinetic + trap − rotation; phân biệt
  điều kiện giam giữ cần thiết với định lý tồn tại nghiệm cho mọi coupling.
- Phân biệt đạo hàm phổ với tích phân kernel bậc hai; không gán hội tụ phổ cho
  toàn bộ phép rời rạc phi địa phương.
- Làm rõ gauge kernel log, chiều dài vật lý a_z và chiều dài không thứ nguyên
  l_z=a_z/a_perp, phép chiếu quintic, và giả thiết frozen Gaussian.
- Sửa TF có quay: bỏ quantum pressure còn cần giả thiết vận tốc rigid-body
  coarse-grained mới suy ra centrifugal potential. TF cho ca ít/không vortex
  chỉ là tham chiếu theo cấu hình, không tự chứng minh nghiệm đúng.
- Phân biệt đối chiếu functional, năng lượng cuối, residual và hội tụ không
  gian. Mười ca độc lập đều dừng line-search trên đích residual 1e-8; giữ nguyên
  trạng thái này dù năng lượng đồng ý dưới 1e-8.
- Bổ sung bảng đối chứng CPU 2×2 shift/BB với toàn bộ kết quả, không chọn riêng
  cấu hình có lợi. Không dùng ca untrapped để tuyên bố ca trapped đã hội tụ.
- Bỏ hình first-run lặp, sửa ví dụ đổi dimension có thể tạo lưới 384^3 và tránh
  hứa vortex count cố định khi thay seed hoặc lịch sử minimizer.
- Sửa DOI Gaidamour, trang Bao2006, thêm nguồn Membrado1989 đã xác minh metadata.
  45 khóa trích dẫn có trong bibliography; không coi kiểm tra metadata là đã
  đọc và xác minh mọi mệnh đề trong mọi bài tham khảo.
- Thêm funding không nhận specific grant theo xác nhận của tác giả; bỏ tuyên
  bố đã lưu CPC Program Library. Thư gửi biên tập không tự xác nhận đồng thuận
  hay tính nguyên gốc trước khi tác giả xác nhận; xác nhận đầy đủ đã nhận ngày 31/10.

## Bằng chứng tính toán

| Phạm vi | Kết quả và giới hạn |
|---|---|
| Release 1.2.2 CPU | 109/109 test đạt, 286.12 s, không skip; chạy lại 29/10 |
| Reference release CPU | 7/7 ca đạt, gate đạt; sai khác năng lượng scaled lớn nhất 2.58115e-11; residual từng ca <1e-6 |
| CUDA rc1 | 97/97 đạt trên T4, 29.52 s; không thay cho 109 test release |
| Batch GPU lịch sử | 56 PASS, 14 REVIEW, 1 FAIL / 71 công việc; giữ nhãn gốc |
| Đối chứng shift/BB | 4 ca × 4 biến thể × 3 lặp = 48; 45 đạt residual, ba iteration cap cùng biến thể quay |
| Tính lại đối chứng | 48/48 trạng thái: năng lượng khớp tuyệt đối trong môi trường này, sai khác residual tối đa 6.16e-19; trace giảm năng lượng |
| CUDA release | Chưa có dữ liệu; notebook 109 test đã khóa nguồn, không suy PASS từ cell có sẵn |
| Trapped-3D refinement | Protocol sáu job đã chuẩn bị, chưa chạy; kết quả trong bài được giới hạn là finite-grid |

Trong đối chứng, shift median + BB không luôn nhanh nhất: ca Newton 3D thô có
fixed shift nhanh hơn, còn tắt BB giảm số lần tính năng lượng ở một số ca. Các
lặp dùng cùng seed, chỉ đánh giá biến thiên thời gian; không phải 12 seed độc
lập. Thời gian đo trên máy dùng chung, không phải benchmark tăng tốc GPU. Ca
3D thô không bổ sung được bằng chứng hội tụ liên tục ở lưới sản xuất.

## Chất lượng mã và tính tái lập

Mã số học 19 module và tám tệp test giữ nguyên byte chuẩn hóa của tag v1.2.2
`1c7eef613d147b6857375f9a61342b38cae1bc86`; không hạ tolerance, sửa kết quả gốc
hay bỏ test. Bản cập nhật Git bổ sung benchmark, công cụ và tài liệu. Script
thực đã dùng cho đối chứng được lưu cùng bằng chứng và hash.

Công cụ hoàn tất không còn checkout --force hoặc đóng gói tiếp sau thất bại
tham chiếu. Verifier phân biệt integrity PASS với evidence/readiness. Kiểm tra
CUDA release khi có dữ liệu đòi đủ 109 ID duy nhất khớp CPU, không failure/error/
skip, backend CUDA, worker và 27 hash nguồn/test phù hợp. Snapshot lịch sử lấy
từ Git cho phép kiểm tra portable; hash gốc được giữ, không giả chứng thư có ký.

Bài chính đã gộp supplementary; PDF có 38 trang, hash bắt đầu 405bfae81f81. Kiểm tra
biên dịch, render và khôi phục gói có log riêng trong `final_revision_2026_10_29`.
Các ZIP được đọc lại để kiểm CRC/hash, không đưa môi trường/cache vào bộ nộp.
`SHA256.json` trong thư mục submission là chỉ mục hash của các gói hoàn chỉnh.

## Phù hợp CPC và rủi ro phản biện

Phần mềm có license MIT, Program Summary, thuật toán, mô hình vật lý, ví dụ,
kiểm thử, đánh giá sai số/hiệu năng và dữ liệu phiên bản hóa: phù hợp hướng
Computer Programs in Physics. Theo phạm vi chính thức, CPC vẫn đòi hỏi chương
trình có ý nghĩa và giải thích rõ tính mới, tầm quan trọng, ứng dụng vật lý;
việc tích hợp nhiều chế độ không tự bảo đảm vượt vòng biên tập. Câu hỏi phản
biện hợp lý còn là lợi ích thực so với solver sẵn có, độ chính xác ca 3D có bẫy
và độ ổn định trên nhiều seed/tham số. Bảng đối chứng mới cải thiện bằng chứng
cho lựa chọn triển khai nhưng không thay thế benchmark đối đầu công bằng với
phần mềm xuất bản khác.

Nguồn chính thức: https://shop.elsevier.com/journals/computer-physics-communications/0010-4655
(đã đọc lại 31/10/2026). Trang Guide for Authors đầy đủ trả 403 trong công cụ;
không tuyên bố đã xác nhận 100% yêu cầu biểu mẫu hiện hành của cổng nộp.

## Những việc còn chờ

1. Khi có GPU, chạy GPU_RELEASE_RETEST; không cần chạy lại 71 job chỉ để bổ sung
   regression release. Trả nguyên ZIP và checksum.
2. Chạy GPU_TRAPPED3D_REFINEMENT để củng cố ca trapped 3D; giữ cả trạng thái OOM,
   timeout hoặc tiêu chí chưa đạt. Nếu chưa làm, tiếp tục giữ giới hạn trong bài.
3. Đã nhận xác nhận của người dùng cho cả hai tác giả trên đúng PDF cuối về
   nội dung, nguyên gốc, không nộp đồng thời, competing interests, CRediT và AI.
4. Hoàn tất lưu/cung cấp program files theo CPC Program Library và kiểm tra các
   mục bắt buộc trên cổng nộp. Chưa có DOI deposit để ghi vào bài.

Đây là đánh giá kỹ thuật và khoa học có giới hạn bằng chứng, không phải quyết
định biên tập hoặc chứng nhận phần mềm đúng cho mọi bài toán.
