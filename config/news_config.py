SITE_CONFIGS = {
    # Xác nhận từ HTML thực tế:
    # - title: h1.title-detail ✓
    # - category: từ breadcrumb "Trở lại Thời sự" → selector: ul.breadcrumb li:nth-child(2) a
    # - description: p.description ✓ (loại bỏ location stamp bên trong nếu có)
    # - content: p.Normal ✓ (nhiều thẻ p.Normal)
    'vnexpress.net': {
        'title': 'h1.title-detail',
        'category': 'ul.breadcrumb li:nth-child(2) a',
        'description': 'p.description',
        'content': 'p.Normal',
    },

    # Xác nhận từ HTML thực tế (user đã fix):
    # - title: h1.detail-title
    # - category: div.detail-top div.detail-cate a (user đã cập nhật)
    # - description: h2.detail-sapo
    # - content: div.detail-content p (bao gồm cả title-content h2)
    'thanhnien.vn': {
        'title': 'h1.detail-title',
        'category': 'div.detail-top div.detail-cate a',
        'description': 'h2.detail-sapo',
        'content': 'div.detail-content p',
    },

    # Tuổi Trẻ dùng layout khác VnExpress dù tên class giống
    # - title: h1.article-title (tuoitre dùng class này, không phải title-detail)
    # - category: div.breadcrumb a:nth-child(2) hoặc a.category-link
    # - description: h2.article-sapo
    # - content: div.article-content p
    'tuoitre.vn': {
        'title': 'h1.article-title',
        'category': 'nav.breadcrumb a:nth-of-type(2)',
        'description': 'h2.article-sapo',
        'content': 'div.article-content p',
    },

    # Dân Trí - xác nhận từ HTML thực tế bài "Cảnh sát khống chế hỏa hoạn":
    # - category: span.category-site-link (breadcrumb line 730: "Thời sự")
    # - title: h1.title-page (hoặc h1[class*="title"])
    # - description: h2.singular-sapo
    # - content: div.singular-content p
    'dantri.com.vn': {
        'title': 'h1.title-page',
        'category': 'div.breadcrumb a:last-child',
        'description': 'h2.singular-sapo',
        'content': 'div.singular-content p',
    },

    # VietnamNet:
    # - title: h1.content-detail-title
    # - category: div.breadcrumb a:last-child
    # - description: h2.content-detail-sapo (hoặc div.content-detail-sapo)
    # - content: div.content-detail-body p
    'vietnamnet.vn': {
        'title': 'h1.content-detail-title',
        'category': 'ul.breadcrumb a:last-of-type',
        'description': 'div.content-detail-sapo',
        'content': 'div.content-detail-body p',
    },

    # Znews (znews.vn - tên miền mới của Zing News):
    # - title: h1.article-title
    # - category: div.breadcrumb a:last-child
    # - description: p.article-summary
    # - content: div.article-content p
    'znews.vn': {
        'title': 'h1.article-title',
        'category': 'ul.breadcrumb li:last-child a',
        'description': 'p.article-summary',
        'content': 'div.article-content p',
    },

    # Giữ lại zingnews.vn phòng trường hợp URL cũ
    'zingnews.vn': {
        'title': 'h1.article-title',
        'category': 'ul.breadcrumb li:last-child a',
        'description': 'p.article-summary',
        'content': 'div.article-content p',
    },
}