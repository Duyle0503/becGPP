# Kiểm tra GPU trong một cell

Mở `GPU_CHECK_ONE_CELL.ipynb` trong Kaggle hoặc Colab, bật GPU rồi chạy cell duy nhất. Hoặc chép toàn bộ `GPU_CHECK_ONE_CELL.py` vào một cell Python. Cell tự tải đúng mã, bổ sung thư viện kiểm tra còn thiếu, chạy từng công việc và tạo ZIP kết quả. Không cần cell cài đặt riêng và không thay PyTorch CUDA đang có.

Mã và bộ test sau sửa được khóa tại commit `75f0a304acf2ff61dceb3c5ac4df10a9ef84b197`, phiên bản 1.2.2rc1. Các module tính toán giống hệt commit `5f66bfc8635a1791078fdd503160e341fb12e82c` đã được đo trên Tesla T4; chỉ test fixture và tài liệu được sửa. Cell kiểm tra SHA256 chuẩn hóa xuống dòng của các module và test trước khi chạy. Nếu notebook đã import một bản becGPP khác, các worker vẫn chạy trong tiến trình mới với đường dẫn nguồn đã khóa.

## Cách chạy

1. Bật GPU và Internet trong môi trường notebook. Nếu chạy offline, đặt `SOURCE_DIR` thành thư mục giải nén đúng commit đã khóa.
2. Giữ `PROFILE = "full"`, `RUN_PAPER = True`, `RUN_METHOD_COMPARISON = True` để chạy đủ 71 công việc. Những ca 3D N=128, nhiều seed và double precision có thể cần nhiều giờ; cần đủ thời gian phiên GPU và dung lượng lưu kết quả.
3. Chạy cell. Mỗi công việc in tiến độ, có heartbeat sau 30 giây, log riêng và giới hạn mặc định 10800 giây/công việc.
4. Tải tệp ZIP có dòng `SEND THIS ZIP BACK`. Trên Kaggle, kết quả nằm dưới `/kaggle/working/`; các môi trường khác dùng thư mục hiện tại.

`PROFILE = "quick"` giảm lưới và ngân sách để thử môi trường, còn 59 công việc và vẫn kiểm tra các nhóm chức năng. Kết quả quick không thay cho số liệu ở lưới của bài. `RUN_PAPER = False` bỏ nhóm showcase nặng; `RUN_METHOD_COMPARISON = False` bỏ 4 so sánh thuật toán. Không bật `ALLOW_CPU_SELFTEST`: tùy chọn này chỉ dùng để kiểm tra bộ điều phối trên máy phát triển và luôn ghi GPU=false.

Riêng 10 ca NumPy/SciPy giữ lưới tham chiếu và đủ 4000 vòng ở cả hai profile: thử nghiệm cho thấy cắt xuống 1200 vòng khiến lattice quay chưa đạt kiểm tra năng lượng. Quick cũng vẫn chạy gate cũ có hội tụ kernel 3D đến N=160; đây không phải chế độ tối thiểu về bộ nhớ.

## Các nhóm kiểm tra

| Nhóm | Nội dung |
|---|---|
| Hồi quy CPU | 97 test cũ chạy nguyên trạng trong tiến trình ẩn CUDA |
| Hồi quy CUDA | Đủ 97 test sau sửa fixture theo thiết bị; không còn loại 7 ca tham chiếu khỏi lượt CUDA |
| Gate giải tích | Gaussian cho các kernel, Landau level, TF và bậc hội tụ; N=512 ở full, N=256 ở quick |
| Kernel trên GPU | Trường và năng lượng Newton 2D/3D, log 2D, q2D so với kernel tích phân độc lập và tổng trực tiếp; padding 2/3; virial q2D ở giới hạn l_z=1e-8 |
| Cực tiểu hóa độc lập | 10 ca: harmonic, quartic, quintic, Newton, log, q2D, CQ tự do, harmonic 3D, Schrödinger–Newton 3D và lattice quay; NumPy/SciPy xây hàm năng lượng riêng, kiểm tra đạo hàm sai phân và L-BFGS |
| CPU–GPU | 4 ca với cùng trạng thái đầu đã lưu: harmonic, q2D, SN 3D và lattice quay; năng lượng, mật độ, sai khác hàm sóng sau chỉnh pha và residual |
| Hội tụ | Newton 2D, log 2D, q2D và SN 3D; 3 lưới ở full, padding 2/3 và hộp lớn hơn ở cùng bước lưới |
| Ca của bài | 8 cấu hình vật lý và lưới lấy từ checkpoint lịch sử; mỗi seed là một công việc riêng. Ca quay có đủ triangular, TF, Gaussian; chọn năng lượng thấp nhất và giữ trạng thái hội tụ thật |
| q2D bổ sung | Ca có bề dày hữu hạn l_z=0.5; ca quasi2D lịch sử dùng giới hạn mỏng 1/r, không được nhầm hai trường hợp |
| Hiệu năng | 2D N=128/256/512; 3D N=32/48/64; CPU và GPU, cùng 80 vòng, warmup, 3 lần đo, đồng bộ CUDA; lưu thời gian và bộ nhớ đỉnh |
| Thời gian đạt độ chính xác | 4 ca harmonic 2D, lattice quay, harmonic 3D, self-gravity 3D: CG preconditioned, CG plain và imaginary-time; cùng seed; ghi thời gian đạt sai khác năng lượng 1e-4, 1e-5, 1e-6 và residual cuối |

Mỗi nghiệm còn được kiểm tra chuẩn khối lượng, dtype float64/complex128, thiết bị thực, năng lượng không tăng ở bước được chấp nhận, đẳng thức hóa thế, virial, khối lượng sát biên và residual. Trạng thái đầu/cuối, ảnh mật độ, trace và cấu hình đầy đủ được lưu để khảo sát lại.

## Cách đọc kết quả

`SUMMARY.json` là điểm bắt đầu. `jobs_summary.csv` là bảng tóm tắt; `outcomes.json` chứa các kết quả chi tiết. Trong `jobs/<tên ca>/` có `stdout.log`, `result.json`, cấu hình, trace và trạng thái khi phù hợp. `source_manifest.json`, `environment.json`, `pip_freeze.txt`, `nvidia_smi.txt` và `SHA256.json` ghi phiên bản, phần cứng và dấu vân tay kết quả.

- `PASS`: đạt điều kiện được nêu cho công việc/nhóm đó.
- `REVIEW`: chạy được nhưng còn điều kiện cần xem, chẳng hạn residual chưa đạt, virial hoặc biên hộp vượt ngưỡng.
- `FAIL`: assertion, sai số kiểm tra hoặc lỗi thực thi; xem log.
- `BLOCKED_MEMORY`, `TIMEOUT`, `INCOMPLETE`: chưa có bằng chứng hoàn tất, không được tính là đạt.

Các ca tham chiếu yêu cầu residual 1e-8. Từ lượt CPU trước, một số ca dừng ở khoảng 1e-8–1e-7 dù năng lượng đồng ý rất tốt; vì vậy kết quả tổng `REVIEW_OR_INCOMPLETE` có thể xuất hiện dù các gate về năng lượng đều đạt. Cell tách `energy_pass` và `strict_residual_pass`, không đổi nhãn để che trạng thái chưa hội tụ.

Ngưỡng nhóm hội tụ được đặt trước: sai khác năng lượng giữa hai lưới cuối <5e-3, padding <2e-6, hộp <5e-3; residual phải đạt và khối lượng sát biên <1e-5. Đây là kiểm tra hữu hạn, không tự chứng minh độ chính xác liên tục của mọi số trong bài. Cần đọc cả xu hướng các lưới và virial.

Ở nhóm showcase, giữ tham số vật lý, lưới, ngân sách và bước tìm kiếm lịch sử; dùng solver sửa mới, tắt dừng theo plateau năng lượng để yêu cầu residual rõ ràng. Đây là phép tính lại với ứng viên, không phải phát lại byte-for-byte số liệu cũ. Một nghiệm có năng lượng thấp nhất trong các seed đã thử vẫn không chứng minh cực tiểu toàn cục.

Nhóm hiệu năng CPU–GPU đo solver cùng ngân sách vòng lặp; nếu solver dừng sớm, tỷ số được để trống. Nhóm so sánh thuật toán riêng chạy lại CG và imaginary-time trên bản ứng viên: năng lượng tham chiếu là thấp nhất trong các trạng thái cuối đã trả về, kèm residual thật; không tự coi đó là nghiệm chính xác hay cực tiểu toàn cục. Nếu không đạt mục tiêu trong ngân sách, thời gian được để trống. Imaginary-time dùng thuật toán tích phân riêng nhưng dùng chung kernel convolution và hàm đo năng lượng của becGPP; đây không phải kiểm chứng độc lập toàn bộ toán tử như nhóm NumPy/SciPy. Chỉ phần lớp thuật toán được trích từ script cũ; tuyệt đối không chạy phần cài phiên bản solver lịch sử. Không ghi đè dữ liệu ITP lịch sử.

Máy chuẩn bị bộ nộp không có CUDA. Cell sửa đã được chạy riêng trên Tesla T4; đủ 97 regression đạt, không skip hay loại test. ZIP full cung cấp bằng chứng cho các phép tính vật lý; ZIP regression riêng bổ sung kết quả sau sửa. Nếu thiếu VRAM, cell ghi rõ và không tự giảm lưới rồi gọi đó là kết quả paper-resolution.

## Kết quả đã nhận và cell kiểm tra lại

Lượt Tesla T4 đầy đủ hoàn tất 71 công việc: 56 PASS, 14 REVIEW, 1 FAIL. Toàn bộ 10 so sánh năng lượng độc lập và 8 trạng thái chính đạt điều kiện tương ứng. Job CUDA cũ có 74/90 test đạt và 16 lỗi tạo tensor CPU/CUDA; 7 ca khác từng bị loại. Sau sửa, CPU và lượt Tesla T4 riêng đều đạt đủ 97 test. Lượt GPU mới mất 29,52 giây, không lỗi/skip/loại test; đã đối chiếu mã, worker, checksum và danh sách test. Không đổi nhãn kết quả cũ thành PASS, không xóa 14 cờ REVIEW vật lý.

`GPU_RETEST_ONLY.ipynb` và `GPU_RETEST_ONLY.py` là một cell tái lập 97 test CUDA đã sửa. Kết quả đã nhận ở `data/raw/gpu_retest_2026_10_17/` của dự án. Một lượt chỉ có 1 job không thay cho bằng chứng của toàn bộ phép tính vật lý; dùng cùng batch gốc và báo cáo đánh giá.
