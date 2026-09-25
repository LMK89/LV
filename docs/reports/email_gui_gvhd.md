# Thư gửi Giảng viên Hướng dẫn tiềm năng

> Các chỗ trong ngoặc vuông `[...]` là thông tin cá nhân, cần điền trước khi gửi.
> Thư chỉ nêu những gì đã làm xong và đã kiểm chứng. Phần chưa có số liệu được
> ghi rõ là "dự kiến".

---

**Tiêu đề:** Xin được Thầy/Cô hướng dẫn Luận văn Thạc sĩ: ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng mô hình thị giác–ngôn ngữ

Kính gửi [Học hàm, học vị, họ tên Thầy/Cô],

Em là [Họ tên], học viên cao học ngành [Ngành], khóa [Khóa], [Trường/Khoa]. Em đã
tìm hiểu các công trình của Thầy/Cô về [lĩnh vực liên quan, ví dụ: xử lý ngôn ngữ
tự nhiên tiếng Việt / nhận dạng văn bản]. Em viết thư này để kính mong Thầy/Cô xem
xét nhận hướng dẫn luận văn của em, với đề tài dự kiến: **"Ràng buộc cấu trúc âm
tiết tiếng Việt cho OCR bằng VLM dùng byte-level BPE: phân tích giới hạn và giải
pháp"**.

**1. Vấn đề nghiên cứu**

Nhiều mô hình thị giác–ngôn ngữ (VLM) hiện nay, tiêu biểu là Florence-2, dùng bộ
tách từ byte-level BPE vốn được xây chủ yếu cho tiếng Anh. Khi áp dụng cho tiếng
Việt, một âm tiết có dấu bị cắt thành nhiều token, và nhiều token trong đó chỉ là
một mảnh byte UTF-8. Trên dữ liệu của em, trung bình một âm tiết tốn khoảng 4 token,
và hơn một nửa số âm tiết bị cắt thành từ 4 token trở lên. Vì mô hình sinh từng mảnh
byte một, nó có thể tạo ra chuỗi byte không hợp lệ (hiển thị thành ký tự U+FFFD),
hoặc những âm tiết không tồn tại trong tiếng Việt.

**2. Hướng tiếp cận**

- *Tinh chỉnh bằng DoRA và chẩn đoán lỗi.* Ở phiên bản thử nghiệm đầu, em quan sát
  thấy 26,7 % ký tự đầu ra là U+FFFD. Em đã truy nguyên được một giả thuyết mạnh:
  adapter được gắn vào lớp `lm_head`, lớp này dùng chung trọng số với embedding, và
  bước gộp adapter khi suy luận đã làm méo embedding đầu vào. Em đã sửa cấu hình
  theo hướng này. Việc xác nhận trực tiếp trên GPU là bước đầu tiên của giai đoạn
  tới.
- *Hàm mất mát nơ-ron–ký hiệu (NeSy Loss).* Đây là một thành phần phạt khả vi, đẩy
  xác suất ra khỏi các token tạo âm tiết không hợp lệ. Hàm có cửa sổ tha thứ ngữ
  cảnh (tối đa 5 token) để không phạt các mảnh của một âm tiết hợp lệ, và không bao
  giờ phạt token đúng của nhãn. Để tách tác dụng của tri thức âm tiết khỏi tác dụng
  điều chuẩn chung, em dùng một nhóm đối chứng là mặt nạ ngẫu nhiên cùng kích thước,
  cùng vị trí phạt.
- *Giải mã có ràng buộc (dự kiến).* Một automat hữu hạn ở cấp byte, chỉ cho phép sinh
  âm tiết hợp lệ, có lối thoát cho tên riêng và từ viết tắt.
- *Đo trần cải thiện.* Em căn chỉnh nhãn và đầu ra ở cấp âm tiết bằng khoảng cách
  Levenshtein có trọng số, có xử lý lỗi tách và gộp từ. Mỗi lỗi được xếp vào một
  trong năm nhóm: âm tiết không hợp lệ, âm tiết không hợp lệ mà nhãn là tên riêng,
  sai dấu, sai chữ cái, và chèn/xóa. Từ đó em tính "CER Oracle", tức mức giảm CER
  tối đa nếu sửa đúng mọi âm tiết không hợp lệ. Con số này là trần lý thuyết cho mọi
  phương pháp ràng buộc âm tiết, và quyết định hướng đi tiếp theo của luận văn.

**3. Giao thức thực nghiệm**

- Dữ liệu gồm 1.350 ảnh dòng văn bản, chia 70/15/15 theo **mã bài** (train 936,
  val 202, test 212 dòng). Không có bài, tài liệu hay câu nào trùng giữa các tập
  (đánh giá độc lập tài liệu).
- Kiểm định ý nghĩa thống kê bằng bootstrap ghép cặp, lấy mẫu lại theo cụm tài liệu
  trên CER toàn corpus, có hiệu chỉnh Holm.
- Các ngưỡng quyết định (trần 2 điểm % CER, bộ giải mã phải đạt 70 % của trần) được
  đăng ký trước, trước khi có số liệu huấn luyện.
- Thí nghiệm chính quét λ ∈ {0; 0,1; 0,5} × {mặt nạ luật, ngẫu nhiên} × 3 seed,
  tổng cộng 15 lượt. Toàn bộ mã nguồn, cấu hình và kiểm thử đã sẵn sàng. Em đang chờ
  tài nguyên GPU để chạy.

**4. Những điểm em mong được Thầy/Cô góp ý**

- Tập test chỉ có 7 bài và 33 tài liệu, nên hiệu ứng khoảng 1–2 điểm CER khó đạt ý
  nghĩa thống kê. Em rất mong được Thầy/Cô chỉ dẫn về hướng mở rộng dữ liệu, hoặc về
  cách báo cáo hiệu ứng nhỏ nhất phát hiện được.
- Các ngưỡng đăng ký trước nêu trên hiện chỉ là đề xuất của em. Em rất cần ý kiến của
  Thầy/Cô trước khi chạy thí nghiệm.

Nếu Thầy/Cô có thời gian, em kính mong được gặp trực tiếp hoặc trao đổi trực tuyến
khoảng 20–30 phút, vào thời điểm thuận tiện cho Thầy/Cô, để trình bày chi tiết và
xin ý kiến định hướng. Em có thể gửi trước bản mô tả kỹ thuật và mã nguồn nếu
Thầy/Cô cần.

Em xin chân thành cảm ơn Thầy/Cô đã dành thời gian đọc thư.

Trân trọng,
[Họ tên]
[Mã học viên] · [Lớp/Khóa] · [Số điện thoại] · [Email]
