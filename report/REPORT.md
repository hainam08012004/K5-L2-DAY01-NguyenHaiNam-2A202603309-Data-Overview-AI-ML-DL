# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:**
   - `class_id` = `468`
   - `class_name` = `cab`
   - `rank` = `1`
   - `score` = `0.510915`
   - `taxonomy_name` = `ImageNet-1K`

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**
   - Record này nói về **cả bức ảnh**.
   - *Vì sao:* Trong cấu trúc dữ liệu của tác vụ Phân loại ảnh (Image Classification), hoàn toàn không có tọa độ không gian (không có bounding box `bbox_xyxy` hay đa giác `polygon_xy`). Mô hình lấy toàn bộ các pixel trong ảnh làm đầu vào và gán một phân phối xác suất lớp cho toàn thể bức ảnh dựa trên đặc trưng nổi bật nhất mà mô hình trích xuất được.

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**
   - Danh sách tên lớp đi kèm với **taxonomy / checkpoint** (cụ thể là danh mục 1.000 lớp cố định của tập dữ liệu ImageNet-1K đi kèm checkpoint `yolo11n-cls.pt`). Mô hình học máy chỉ gán điểm xác suất cho các lớp có sẵn trong taxonomy này chứ không thể tự nghĩ ra lớp mới.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**
   - `class_name` giúp con người đọc và hiểu trực quan ngữ nghĩa.
   - `class_id` là mã số kỹ thuật để máy tính truy xuất và lập trình xử lý.
   - `taxonomy_name` xác định hệ quy chiếu và không gian lớp (ImageNet-1K).
   - Nếu chỉ ghi mỗi tên lớp thì rất dễ bị nhập nhằng ngữ nghĩa (ví dụ `cab` vs `taxi`), hoặc cùng một `class_id` nhưng ở taxonomy khác lại trỏ vào một vật thể hoàn toàn khác. Ghi đủ cả 3 trường giúp đảm bảo tính truy xuất nguồn gốc (provenance), tránh sai sót khi tích hợp vào hệ thống dữ liệu.

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**
   - Guideline cần đưa ra **quy tắc ưu tiên rõ ràng (priority rule)**:
     - Ưu tiên đối tượng chiếm diện tích lớn nhất (dominant object) hoặc nằm ở vị trí trung tâm bức ảnh.
     - Ưu tiên theo mục tiêu nghiệp vụ của dự án (ví dụ: trong bài toán giao thông, ưu tiên phương tiện đang chuyển động hơn là cảnh nền đường phố/tòa nhà).
     - Quy định cụ thể khi nào cần chuyển sang bài toán gán đa nhãn (multi-label classification) nếu bức ảnh có nhiều chủ đề ngang nhau mà không thể ép về một nhãn duy nhất.

6. **Vì sao không được lấy `score` của model làm ground truth?**
   - Vì `score` (confidence) chỉ là độ tin cậy toán học nội tại của mô hình dựa trên trọng số đã học, hoàn toàn **không phản ánh sự thật khách quan (ground truth)**.
   - Mô hình có thể có `score` rất cao nhưng vẫn đoán sai (ví dụ cảnh giao thông đông đúc hỗn hợp nhưng AI lại đoán cả ảnh là `cab`), hoặc ngược lại score thấp nhưng nhãn vẫn đúng. Ground truth bắt buộc phải là nhãn do con người thẩm định và xác nhận theo guideline chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record:**
   - `class_name` = `person`
   - `score` = `0.912625`
   - `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`
   - `bbox_width` = `113.58`
   - `bbox_height` = `279.68`

2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
   - Trong hệ tọa độ pixel của ảnh (gốc tọa độ `(0, 0)` nằm ở góc trên cùng bên trái):
     - Góc trên-trái của hộp nằm tại tọa độ pixel `(385.33, 69.24)`.
     - Góc dưới-phải của hộp nằm tại tọa độ pixel `(498.92, 348.92)`.
     - Kích thước hộp: Chiều rộng là `113.58 pixel` và chiều cao là `279.68 pixel`, bao quanh người đàn ông đeo tạp dề đang đứng quay lưng lại ở gian bếp.

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
   - Tại ngưỡng `0.60`: Có **6 prediction** được giữ lại (chỉ gồm các vật thể mô hình có độ tự tin cao: 2 person, 2 bowl, 2 oven).
   - Tại ngưỡng chuẩn `0.35`: Có **11 prediction** (bổ sung thêm các cốc `cup` và bát `bowl` nhỏ trên quầy bếp).
   - Tại ngưỡng `0.20`: Có nhiều prediction hơn (khoảng 14–16 vật thể, giữ lại cả các vật thể mờ hoặc độ tin cậy thấp).
   - *Nhận xét:* Ngưỡng càng thấp thì số lượng hộp được giữ lại càng nhiều nhưng tỷ lệ dự đoán sai càng tăng; ngưỡng càng cao thì lọc càng chặt chẽ nhưng dễ bỏ sót vật thể nhỏ.

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Khi hạ ngưỡng, mô hình sẽ sinh ra rất nhiều hộp dương tính giả (false positives - nhận nhầm họa tiết, nền thành đồ vật) hoặc vẽ nhiều hộp trùng lặp nhau (như cụm `cup 0.45` và `cup 0.38` đè lấn lên nhau trên bàn bếp).
   - Người **Reviewer** sẽ phải:
     1. Tốn nhiều thời gian và công sức hơn để duyệt qua từng hộp trên ảnh.
     2. Xóa bỏ (reject/delete) các hộp rác, hộp đoán sai nhãn.
     3. Tinh chỉnh kích thước hộp hoặc hợp nhất các hộp bị trùng lặp trên cùng một vật thể.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):**
   - **Quy tắc đóng khung ôm sát (Tight Bounding Box):**
     *"Khung chữ nhật phải ôm khít 4 điểm cực biên (trên, dưới, trái, phải) của vật thể, không để khoảng trống thừa quá 2–3 pixel so với vật thể thực và tuyệt đối không được cắt lẹm vào phần thân của vật thể. Nếu vật thể có phần nhô ra (như cánh tay người, tay cầm chảo/nồi), khung bao phải bao trùm cả phần nhô ra đó."*

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - *Quan sát trên ảnh:*
     - Chiếc bếp nướng ở mép trái (`oven 0.69`) bị cắt mép sát cạnh ảnh (`x_min = 0.11 px`).
     - Ở góc dưới bên trái có một hộp `person 0.61` (tọa độ `[0.12, 263.18, 61.55, 310.78]`) thực chất chỉ là một cánh tay/bàn tay thò vào mép ảnh.
   - *Cách xử lý:*
     - Với vật thể bị cắt mép thông thường (như chiếc bếp `oven 0.69`): Viết thành quy tắc trong **Guideline** (*"Đóng khung sát mép ảnh chụp tại x=0 và đánh dấu thuộc tính is_truncated = True"*). Đây là hiện tượng chụp ảnh thông thường, không cần hỏi mỗi lần gặp.
     - Với trường hợp như `person 0.61` (chỉ nhìn thấy mỗi bàn tay/cổ tay tách rời ở mép ảnh): Cần đưa lên **Escalation (hỏi Lab Coach)**: *"Khi chỉ nhìn thấy một bộ phận nhỏ tách rời ở mép ảnh mà không thấy thân hay đầu người thì có gán nhãn person hay không?"*. Việc này cần thống nhất ở cấp quản lý dự án để tránh tình trạng annotator này thì gán nhãn, annotator khác lại bỏ qua gây mâu thuẫn dữ liệu.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record:**
   - `instance_id` = `kitchen-001`
   - `class_name` = `person`
   - `score` = `0.899318`
   - `polygon_point_count` = `348`
   - 3–5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - Bounding box (hộp chữ nhật) chỉ cho biết phạm vi kích thước bao ngoài cực đại (extent) và vị trí thô, luôn bị lẫn các điểm ảnh nền (background) hoặc vật thể khác ở các góc hộp.
   - Polygon (đa giác) cung cấp thông tin hình học chi tiết ở cấp độ pixel (pixel-level contour): mô tả chính xác đường cong uốn lượn thực tế của vật thể (như đường viền cánh tay, vạt tạp dề, khe hở giữa hai chân), tách bạch hoàn toàn phần thân đối tượng với hậu cảnh.

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - `instance_id` (ví dụ `kitchen-001`, `kitchen-009`) dùng để định danh duy nhất cho từng cá thể (instance) riêng biệt trong phạm vi một bức ảnh.
   - *Không phải class ID:* Vì class ID là mã của loại đối tượng (cả 2 người trong ảnh đều có `class_id = 0` thuộc lớp `person`, nhưng được gán 2 `instance_id` khác nhau là `kitchen-001` và `kitchen-009` để phân biệt 2 cá thể).
   - *Không phải tracking ID:* Vì tracking ID dùng trong xử lý video để theo vết cùng một vật thể di chuyển liên tục qua nhiều khung hình (frames) theo thời gian. Ở đây là ảnh tĩnh đơn lẻ, `instance_id` chỉ tồn tại để phân biệt các đối tượng độc lập trong bức ảnh đó, không có liên kết thời gian giữa các ảnh.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo:**
   - **Quy tắc bám sát đường biên thực (Polygon Boundary Contour):**
     *"Đường viền đa giác phải ôm khít theo ranh giới pixel thực tế của vật thể với sai số không quá 1–2 pixel. Tại các đoạn thẳng hoặc cong đều, đặt các điểm neo cách nhau hợp lý; tại các nếp gấp, chi tiết lồi lõm hoặc góc nhọn phức tạp (như ngón tay, nếp áo), annotator phải tăng mật độ điểm neo để bảo toàn hình dạng thực, tuyệt đối không được cắt góc (shortcut) làm mất chi tiết hình học của vật thể."*

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - *Quan sát trên ảnh:*
     - Chiếc bàn gỗ (`dining table 0.58`) bị hàng loạt chiếc bát (`bowl 0.74`, `bowl 0.70`, `bowl 0.53`) và dụng cụ trên bàn che khuất một phần bề mặt.
     - Chùm thảo mộc khô treo ở góc trên bên trái bị nhận nhầm thành `potted plant 0.63` và chạm vào các thanh treo/nồi đồng.
   - *Cách xử lý:*
     - Với trường hợp vật thể che khuất nhau trên mặt bàn (bát đè lên bàn `dining table`): Dùng **Guideline**. Vì đây là hiện tượng che lấp (occlusion) rất phổ biến trong phân đoạn ảnh. Guideline chỉ cần nêu quy tắc phân lớp rõ ràng: *"Vật thể ở lớp trên (foreground như bát bowl) được vẽ trọn vẹn biên; vật thể nền ở lớp dưới (như mặt bàn dining table) phải được khoét lỗ/trừ đi phần diện tích bị che khuất"*.
     - Với trường hợp chùm thảo mộc khô bị gán nhãn `potted plant`: Cần đưa lên **Escalation (hỏi Lab Coach)**: *"Chùm lá khô treo trần có thuộc taxonomy potted plant không, hay cần bỏ qua để tránh gây nhiễu cho mô hình học cây cảnh?"*.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Một nhãn danh mục duy nhất (`class_id`, `class_name`, `taxonomy_name`) gán cho **toàn bộ bức ảnh**. | Ảnh `traffic` là đường phố đông đúc gồm nhiều xe buýt, xe hơi, người đi bộ nhưng mô hình đơn nhãn chỉ đoán `cab 0.51` cho cả ảnh; khó chọn 1 nhãn khi ảnh có nhiều chủ đề ngang nhau. | Quan sát toàn ảnh, áp dụng quy tắc ưu tiên trong Guideline (ưu tiên vật chiếm diện tích lớn nhất hoặc ở trung tâm) để chọn **MỘT** nhãn phù hợp nhất. | Kiểm tra nhãn được chọn có tuân thủ đúng quy tắc ưu tiên trong guideline không; phát hiện các ca gắn nhãn chủ quan hoặc ảnh đa chủ đề cần escalation. |
| **Phát hiện vật thể** | Danh sách các khung chữ nhật `bbox_xyxy` (kèm tên lớp) cho **từng vật thể riêng biệt** trong ảnh. | Mô hình nhận nhầm bàn tay thò vào mép trái thành người (`person 0.61`); bỏ sót nhiều xoong nồi treo trên trần; các hộp cốc/bát đè lấn lên nhau trên bàn (`cup 0.45`, `cup 0.38`). Người đứng bếp đầu tiên vẽ hộp bao (`person 0.91`) cũng chưa thật chuẩn, viền bo còn khá bị thừa khoảng trống so với thân người. | Vẽ khung chữ nhật ôm khít 4 điểm cực biên của từng vật thể (Tight Box); xử lý vật thể cắt mép ảnh chạm viền `x=0`; gán thuộc tính che khuất (`is_occluded`) nếu có. | Soát độ khít của khung (phát hiện các khung bị bo thừa khoảng trống hoặc cắt lẹm vào vật); xóa bỏ các hộp rác (dương tính giả); phát hiện vật thể bị bỏ sót và gộp các hộp bị vẽ trùng. |
| **Instance segmentation** | Danh sách các đa giác mặt nạ khép kín `polygon_xy` (kèm `instance_id` và tên lớp) cho **từng cá thể riêng biệt**. | Mô hình nhận nhầm chùm thảo mộc khô treo trần thành `potted plant 0.63`; mặt nạ bàn bếp `dining table` bị khoét lỗ lộn xộn do đồ vật đè lên; viền người bị lấn vào hậu cảnh tối hoặc bo viền còn chỗ bị lỏng/thừa. | Chấm các điểm neo polygon men theo đường viền thực tế của vật thể; tách rời từng instance kể cả khi cùng lớp; khoét lỗ phần bị che lấp khi các vật thể đè lên nhau. | Phóng to kiểm tra độ chính xác của đường viền (không cắt góc lười, không lem/thừa viền vào nền); kiểm tra ranh giới phân tách giữa các vật chạm nhau và tính duy nhất của `instance_id`. |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu):**
   - Chỉ sử dụng 3 ảnh COCO công khai có sẵn trong bài lab, tuyệt đối không tải ảnh cá nhân, khuôn mặt, biển số xe, thông tin định danh hay dữ liệu nội bộ lên Colab hoặc GitHub.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?:**
   - Dừng ngay mọi thao tác xử lý, không tải về hoặc lan truyền dữ liệu đó, và báo cáo ngay lập tức cho Lab Coach / Giảng viên phụ trách để được hướng dẫn cô lập và xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
