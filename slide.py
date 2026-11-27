from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Khởi tạo presentation (Tỷ lệ 16:9)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Định nghĩa màu sắc (Phong cách Steam / Gaming Dark Theme)
BG_COLOR = RGBColor(27, 40, 56)      # Xanh đen Steam tối
CARD_COLOR = RGBColor(42, 71, 94)    # Xanh xám nhạt hơn cho card
TEXT_LIGHT = RGBColor(255, 255, 255) # Trắng
TEXT_MUTED = RGBColor(199, 213, 224) # Xám nhạt
ACCENT_COLOR = RGBColor(102, 192, 244) # Xanh dương sáng Steam

def set_slide_background(slide):
    # Tạo nền tối toàn slide
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR

def add_header(slide, title_text, category_text="STEAM REC-SYS PROJECT"):
    set_slide_background(slide)
    
    # Thêm tiêu đề danh mục nhỏ phía trên
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(10), Inches(0.4))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(12)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_COLOR
    
    # Tiêu đề chính của slide
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.9), Inches(11.5), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(28)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_LIGHT

# ----------------- SLIDE 1: TRANG BÌA -----------------
slide_layout = prs.slide_layouts[6] # Blank layout
slide1 = prs.slides.add_slide(slide_layout)
set_slide_background(slide1)

# Hộp chứa thông tin bìa
title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.5))
tf = title_box.text_frame
tf.word_wrap = True

p0 = tf.paragraphs[0]
p0.text = "STEAM RECOMMENDER SYSTEM"
p0.font.size = Pt(16)
p0.font.bold = True
p0.font.color.rgb = ACCENT_COLOR
p0.space_after = Pt(10)

p1 = tf.add_paragraph()
p1.text = "Hệ thống Gợi ý Trợ chơi Điện tử Thông minh"
p1.font.size = Pt(40)
p1.font.bold = True
p1.font.color.rgb = TEXT_LIGHT
p1.space_after = Pt(15)

p2 = tf.add_paragraph()
p2.text = "Ứng dụng Học tương phản (Contrastive Learning) & Lai ghép (Hybrid) | Team 3 người"
p2.font.size = Pt(18)
p2.font.color.rgb = TEXT_MUTED

# ----------------- HÀM TẠO SLIDE DẠNG CARD BỐ CỤC -----------------
def add_card_slide(title, cards_data):
    """
    cards_data là list các dict: [{'title': '...', 'content': ['...', '...']}, ...]
    Hỗ trợ tối đa 3-4 card trên 1 slide.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, title)
    
    num_cards = len(cards_data)
    total_width = 11.7
    gap = 0.4
    card_width = (total_width - (num_cards - 1) * gap) / num_cards
    card_height = 4.8
    top_pos = 1.8
    
    for i, data in enumerate(cards_data):
        left_pos = 0.8 + i * (card_width + gap)
        
        # Vẽ khung card nền
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left_pos), Inches(top_pos), Inches(card_width), Inches(card_height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_COLOR
        shape.line.color.rgb = ACCENT_COLOR
        shape.line.width = Pt(1)
        
        # Thêm text vào card
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.3)
        tf.margin_left = Inches(0.3)
        tf.margin_right = Inches(0.3)
        
        # Tiêu đề card
        p_head = tf.paragraphs[0]
        p_head.text = data['title']
        p_head.font.size = Pt(18)
        p_head.font.bold = True
        p_head.font.color.rgb = ACCENT_COLOR
        p_head.space_after = Pt(14)
        
        # Nội dung trong card
        for item in data['content']:
            p_item = tf.add_paragraph()
            p_item.text = f"• {item}"
            p_item.font.size = Pt(14)
            p_item.font.color.rgb = TEXT_LIGHT
            p_item.space_after = Pt(8)

# ----------------- TẠO CÁC SLIDE NỘI DUNG -----------------

# Slide 2: Mục tiêu & Tính thực tế
add_card_slide("Bối cảnh & Mục tiêu Đề tài", [
    {
        'title': 'Vấn đề thực tế',
        'content': [
            'Nền tảng Steam có hàng chục nghìn tựa game.',
            'Người dùng bị quá tải thông tin (Information Overload).',
            'Khó khăn trong việc tìm kiếm tựa game phù hợp với gu cá nhân.'
        ]
    },
    {
        'title': 'Mục tiêu hệ thống',
        'content': [
            'Xây dựng công cụ gợi ý game thông minh, cá nhân hóa.',
            'Giải quyết triệt để bài toán Cold-start cho game mới ra mắt.',
            'Hoàn thiện hệ thống thực tế trong vòng 1-2 tháng với team 3 người.'
        ]
    }
])

# Slide 3: Dataset
add_card_slide("Cấu trúc Bộ dữ liệu (Dataset)", [
    {
        'title': 'games.csv & Metadata',
        'content': [
            'Chứa thông tin chi tiết: giá, ngày phát hành, hệ điều hành (Win/Mac/Linux).',
            'Kèm theo tệp metadata chứa mô tả chi tiết (description) và các tags thể loại.'
        ]
    },
    {
        'title': 'recommendations.csv',
        'content': [
            'Lịch sử đánh giá thực tế từ người dùng.',
            'Cung cấp trạng thái đề xuất (is_recommended), số giờ chơi (hours), và độ hữu ích.'
        ]
    },
    {
        'title': 'users.csv',
        'content': [
            'Thông tin công khai của người dùng.',
            'Thống kê tổng số sản phẩm đã sở hữu và số lượng review đã đăng trên nền tảng.'
        ]
    }
])

# Slide 4: Pipeline
add_card_slide("Luồng Xử lý Hệ thống (Pipeline)", [
    {
        'title': '1. Offline Pipeline (Huấn luyện)',
        'content': [
            'Tiền xử lý: Làm sạch dữ liệu, xử lý giá trị null ở metadata.',
            'Feature Engineering: Chuyển đổi số giờ chơi & trạng thái review thành Implicit Rating.',
            'Training: Huấn luyện Deep Learning kết hợp Học tương phản (Contrastive Learning), xuất file trọng số (.pt).'
        ]
    },
    {
        'title': '2. Online Pipeline (Phục vụ)',
        'content': [
            'Serving: Tải file trọng số mô hình đã train.',
            'Inference: Kết hợp tầng Lai ghép (Hybrid Layer) để xử lý yêu cầu thời gian thực.',
            'Demo: Hiển thị danh sách game gợi ý tối ưu qua giao diện Web Streamlit.'
        ]
    }
])

# Slide 5: Kỹ thuật áp dụng
add_card_slide("Các Kỹ thuật RecSys Cốt lõi", [
    {
        'title': 'Hành vi & Cộng đồng',
        'content': [
            'Session-based: Bắt ngay hành vi click game tức thời của khách vãng lai.',
            'Collaborative Filtering: Khai thác ma trận tương tác User-Game dựa trên lịch sử giờ chơi.'
        ]
    },
    {
        'title': 'Nội dung & Tri thức',
        'content': [
            'Content-based & Knowledge Graph: Khai thác tags, mô tả và đồ thị tri thức xử lý game mới.',
            'Contrastive Learning: Tối ưu hóa không gian vector embedding cho Deep Learning.'
        ]
    },
    {
        'title': 'Tầng Tổng hợp',
        'content': [
            'Hybrid Fusion: Tổng hợp điểm số thông minh từ nhiều mô hình đơn lẻ trước khi trả kết quả.'
        ]
    }
])

# Slide 6: Phân công team
add_card_slide("Phân công Công việc (Team 3 người)", [
    {
        'title': 'Thành viên 1',
        'content': [
            'Phụ trách xử lý dữ liệu thô và làm sạch.',
            'Xây dựng Đồ thị tri thức (Knowledge Graph).',
            'Lập trình module Gợi ý dựa trên Nội dung (Content-based).'
        ]
    },
    {
        'title': 'Thành viên 2',
        'content': [
            'Xử lý ma trận tương tác từ file recommendations.',
            'Huấn luyện mô hình Deep Learning kết hợp Học tương phản.',
            'Xuất và quản lý file trọng số mô hình (.pt).'
        ]
    },
    {
        'title': 'Thành viên 3',
        'content': [
            'Xây dựng module Gợi ý theo phiên (Session-based).',
            'Lập trình tầng Lai ghép (Hybrid Score Layer).',
            'Hoàn thiện giao diện Web Demo (Streamlit).'
        ]
    }
])

# Slide 7: Tổng kết & Q&A
slide_end = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_background(slide_end)
end_box = slide_end.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.3), Inches(3.5))
tf_end = end_box.text_frame
tf_end.word_wrap = True

p_e1 = tf_end.paragraphs[0]
p_e1.text = "TỔNG KẾT & Q&A"
p_e1.font.size = Pt(36)
p_e1.font.bold = True
p_e1.font.color.rgb = TEXT_LIGHT
p_e1.space_after = Pt(15)

p_e2 = tf_end.add_paragraph()
p_e2.text = "Sản phẩm: Web Demo hoàn chỉnh tích hợp đầy đủ các kỹ thuật học thuật hiện đại."
p_e2.font.size = Pt(18)
p_e2.font.color.rgb = ACCENT_COLOR
p_e2.space_after = Pt(10)

p_e3 = tf_end.add_paragraph()
p_e3.text = "Cảm ơn Thầy Cô và các bạn đã lắng nghe. Chúng tôi rất mong nhận được góp ý!"
p_e3.font.size = Pt(16)
p_e3.font.color.rgb = TEXT_MUTED

# Lưu file PowerPoint có thiết kế
output_path = "Steam_Recommender_System_Styled.pptx"
prs.save(output_path)
print(f"Đã tạo file PowerPoint có thiết kế layout thành công: {output_path}")