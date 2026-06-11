# 🧩 XẾP TRANH TÍCH HỢP AI VÀ THUẬT TOÁN TÌM KIẾM HEURISTIC

Ứng dụng trực quan hóa trò chơi xếp hình trượt (Sliding Puzzle) lưới $3\times3$ và $4\times4$. Dự án tập trung vào việc đối sánh hiệu năng định lượng và minh họa cây không gian tìm kiếm giữa các giải thuật AI thông dụng.

🚀 **Trải Nghiệm Trực Tuyến:** [Hugging Face Spaces - nhapmonAI](https://huggingface.co/spaces/)

---

## 📌 Giới thiệu

Dự án tập trung nghiên cứu và xây dựng chương trình giải bài toán trò chơi trượt ô số/xếp tranh kinhдени (**8-Puzzle** và **15-Puzzle**) bằng các phương pháp tìm kiếm không gian trạng thái (State-space Search) tích hợp tri thức chuyên gia trên môi trường Python.

Bài toán được tiếp cận dưới dạng **Tìm kiếm Heuristic (Heuristic Search)** trong Trí tuệ nhân tạo truyền thống, đối mặt trực tiếp với thách thức bùng nổ tổ hợp trạng thái. Mô hình đề xuất kết hợp:
* **Thuật toán tìm kiếm thông minh (A\*, IDA\*)** phối hợp cùng thuật toán tìm kiếm mù (**BFS**) để tìm ra chuỗi hành động dịch chuyển ô trống tối ưu.
* **Các dạng hàm đánh giá Heuristic từ cơ bản đến nâng cao** nhằm cắt tỉa nhánh cây tìm kiếm, giảm thiểu số lượng bản ghi cần duyệt và tối ưu hóa bộ nhớ RAM.
* Giao diện trực quan hóa tương tác thời gian thực bằng **Gradio** và đồ họa **Matplotlib**.

---

## 🎯 Mục tiêu

* **Quản lý và vận hành trò chơi:** Khởi tạo ma trận động $3\times3$ và $4\times4$, xáo trộn trạng thái bảo đảm tính khả giải, hoán đổi ô trống theo luật định.
* **Tích hợp tri thức Heuristic:** Triển khai cấu trúc tính toán sai lệch không gian nhằm định hướng thuật toán đi đúng hướng.
* **Đối sánh và Đánh giá thuật toán thông qua:**
  * **Số lượng Node đã duyệt (Nodes Expanded):** Đánh giá độ hiệu quả cấu trúc cây.
  * **Thời gian thực thi (Execution Time - ms):** Đánh giá tốc độ xử lý phần mềm.
  * **Độ dài lời giải (Path Length):** Đánh giá tính tối ưu của đường đi.
* **Trực quan hóa đồ họa:**
  * Cây không gian trạng thái suy luận của AI (Search Tree Graph).
  * Biểu đồ cột so sánh hiệu năng tài nguyên định lượng giữa các thuật toán.
  * Tự động hiển thị và trình diễn bước đi (Auto-Play Animation).

---

## 🛠️ Thư viện sử dụng

| Thư viện | Vai trò |
| :--- | :--- |
| **Gradio** | Xây dựng giao diện Web tương tác trực quan cho người dùng, xử lý sự kiện kích hoạt và vòng lặp thời gian (Timer). |
| **Matplotlib** | Vẽ cây không gian tìm kiếm thực tế và biểu đồ cột đối sánh hiệu năng định lượng. |
| **Pillow (PIL)** | Xử lý hình ảnh, tự động cắt ghép, phân chia nhỏ ảnh tải lên thành các mảnh lưới ma trận trò chơi. |
| **NumPy** | Xử lý ma trận dữ liệu trạng thái bàn cờ trượt một cách nhanh chóng. |
| **Heapq** | Cấu trúc dữ liệu hàng đợi ưu tiên (Priority Queue) tối ưu độ phức tạp $O(\log n)$ cho giải thuật A\*. |

---

## ✨ Tính Năng Chính

* **Tải ảnh tùy chỉnh:** Tự động cắt và chia nhỏ bất kỳ ảnh nào người dùng tải lên thành các ô lưới mảnh ghép hoàn chỉnh tương ứng với kích thước bàn cờ.
* **4 Hàm Heuristic $H(n)$ mạnh mẽ:**
  * *Misplaced Tiles:* Đếm số ô sai vị trí.
  * *Manhattan Distance:* Tổng khoảng cách di chuyển theo trục ngang/dọc.
  * *Euclidean Distance:* Khoảng cách đường chim bay hình học.
  * *Linear Conflict:* Phát hiện và phạt điểm các ô ngáng đường nhau trên cùng hàng/cột (Tối ưu nhất).
* **Đối sánh thuật toán:** So sánh trực quan hiệu năng chạy thực tế giữa ba giải thuật cốt lõi **A\***, **IDA\***, và **BFS**.
* **Đồ thị thời gian thực:** Xuất ảnh cây trạng thái tìm kiếm (Search Tree) động và biểu đồ cột so sánh độ phức tạp.
* **Auto-Play:** Chế độ AI tự động giải và hoạt họa trình diễn trượt các ô trên bàn cờ dựa vào xung nhịp ngắt thời gian.

---

## ⚠️ Những khó khăn gặp phải

Trong quá trình thực hiện nghiên cứu, dự án đối mặt với một số rào cản kỹ thuật:
* **Bùng nổ tổ hợp (Combinatorial Explosion):** Khi chuyển sang cấu hình lưới $4\times4$, không gian trạng thái tăng lên cực kỳ khủng khiếp ($16! \approx 2 \times 10^{13}$). Các thuật toán lưu vết như BFS hoặc A\* dùng Heuristic yếu rất dễ gây ra hiện tượng **tràn bộ nhớ (Out of Memory)**.
* **Giới hạn đồ họa cây:** Số lượng node sinh ra trong cây tìm kiếm thực tế quá lớn, việc vẽ toàn bộ cấu trúc cây lên màn hình gây nghẽn luồng xử lý đồ họa, bắt buộc phải thực hiện cắt tỉa giới hạn số cạnh hiển thị ban đầu.
* **Đồng bộ hóa Animation:** Việc đồng bộ luồng xử lý bất đồng bộ của thuật toán với bộ giao diện Gradio đòi hỏi kiểm soát chặt chẽ trạng thái biến toàn cục.

---

## 🔮 Hướng phát triển

* Triển khai thuật toán nâng cao **Pattern Databases Heuristic** (Tính toán trước dữ liệu) để giải quyết triệt để lưới lớn $4\times4$ và $5\times5$ trong thời gian dưới 1 giây.
* Áp dụng giải thuật **Tìm kiếm hai hướng (Bi-directional Search)** để thu hẹp không gian duyệt cây từ hai đầu trạng thái gốc và đích.
* Tối ưu hóa các hàm tính toán khoảng cách bằng mã máy **Numba** hoặc liên kết thư viện **C++** nhằm đẩy cao tốc độ xử lý vòng lặp.

---

## 💻 3. Cài đặt và chạy dự án

### Clone Repository

```bash
git clone [https://github.com/kamenriderkuuga7805-arch/sliding_puzzle_ai_HUS.git](https://github.com/kamenriderkuuga7805-arch/sliding_puzzle_ai_HUS.git)
cd sliding_puzzle_ai_HUS
