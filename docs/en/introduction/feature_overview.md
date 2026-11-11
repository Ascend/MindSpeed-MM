# Feature Overview

[To be completed]

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Overview | MindSpeed MM Developer Guide</title>
    <style>
        /* Styles extracted from the MindSpeed MM documentation */
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #1abc9c;
            --border-color: #e0e0e0;
            --header-bg: #f8f9fa;
            --row-even-bg: #fafafa;
            --text-primary: #333;
            --text-secondary: #666;
            --bg-light: #f5f7fa;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background-color: var(--bg-light);
            padding: 0;
        }
        .document-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .document-header {
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 6px;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-left: 4px solid var(--secondary-color);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .document-header h1 {
            font-size: 2.2rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 10px;
        }
        .document-header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 800px;
        }
        .breadcrumb {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 15px;
        }
        .breadcrumb a {
            color: var(--secondary-color);
            text-decoration: none;
        }
        .breadcrumb a:hover {
            text-decoration: underline;
        }
        .breadcrumb span {
            margin: 0 5px;
        }
        /* Table styles - modeled after the MindSpeed MM documentation */
        .table-wrapper {
            margin: 30px 0;
            overflow-x: auto;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background-color: white;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            min-width: 700px;
        }
        /* Table header styles - add a gradient background color */
        thead {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
        }
        th {
            padding: 14px 16px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #1a252f;
            border-right: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 0.95rem;
            position: relative;
        }
        th:last-child {
            border-right: none;
        }
        /* Divider lines between header columns */
        th:not(:last-child)::after {
            content: "";
            position: absolute;
            right: 0;
            top: 20%;
            height: 60%;
            width: 1px;
            background-color: rgba(255, 255, 255, 0.3);
        }
        /* Add a vertical line between the "Feature Name" and "Status" columns */
        th:nth-child(3),
        td:nth-child(3) {
            border-right: 2px solid var(--border-color);
            position: relative;
        }
        /* Add vertical line decoration */
        th:nth-child(3)::after,
        td:nth-child(3)::after {
            content: "";
            position: absolute;
            right: -2px;
            top: 20%;
            height: 60%;
            width: 1px;
            background-color: var(--border-color);
        }
        th:nth-child(3)::after {
            background-color: rgba(255, 255, 255, 0.5);
        }
        /* Table body styles */
        td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            vertical-align: top;
        }
        td:last-child {
            border-right: none;
        }
        tbody tr:last-child td {
            border-bottom: none;
        }
        /* Alternating row background color */
        tbody tr:nth-child(even) {
            background-color: var(--row-even-bg);
        }
        /* Row hover effect */
        tbody tr:hover {
            background-color: rgba(52, 152, 219, 0.05);
        }
        /* Cell content styles */
        .category-cell {
            font-weight: 600;
            color: var(--primary-color);
            background-color: rgba(52, 152, 219, 0.03);
        }
        .subcategory-cell {
            color: var(--primary-color);
            font-weight: 500;
            background-color: rgba(52, 152, 219, 0.02);
        }
        .feature-name {
            color: var(--text-primary);
        }
        .empty-cell {
            color: #999;
            font-style: italic;
        }
        .status-cell {
            text-align: center;
        }
        .status-released {
            display: inline-block;
            padding: 5px 12px;
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            color: #2e7d32;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid #a5d6a7;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        /* Category label */
        .category-label {
            display: inline-block;
            padding: 3px 8px;
            background-color: rgba(52, 152, 219, 0.1);
            color: var(--secondary-color);
            border-radius: 3px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-left: 5px;
        }
        /* Footnote area */
        .footnote {
            margin-top: 30px;
            padding: 25px;
            border-top: 1px solid var(--border-color);
            font-size: 0.9rem;
            color: var(--text-secondary);
            background-color: white;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        }
        .footnote-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--primary-color);
            font-size: 1rem;
        }
        /* Responsive design */
        @media (max-width: 768px) {
            .document-container {
                padding: 20px 15px;
            }
            .document-header {
                padding: 20px;
            }
            .document-header h1 {
                font-size: 1.8rem;
            }
            .table-wrapper {
                margin: 20px -15px;
                border-left: none;
                border-right: none;
                border-radius: 0;
                box-shadow: none;
            }
            th, td {
                padding: 10px 12px;
                font-size: 0.9rem;
            }
            /* Adjust the vertical line for mobile devices */
            th:nth-child(3)::after,
            td:nth-child(3)::after {
                top: 15%;
                height: 70%;
            }
        }
        @media (max-width: 480px) {
            .document-header h1 {
                font-size: 1.6rem;
            }
            th, td {
                padding: 8px 10px;
            }
            .status-released {
                padding: 4px 10px;
                font-size: 0.8rem;
            }
        }
        /* Document link styles */
        .doc-links {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }
        .doc-link {
            display: flex;
            align-items: center;
            padding: 10px 15px;
            background-color: white;
            border-radius: 4px;
            text-decoration: none;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }
        .doc-link:hover {
            border-color: var(--secondary-color);
            box-shadow: 0 2px 6px rgba(52, 152, 219, 0.1);
            transform: translateY(-2px);
        }
        .doc-link-icon {
            margin-right: 10px;
            color: var(--secondary-color);
        }
    </style>
</head>
<body>
    <div class="document-container">
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th width="20%">Feature Category</th>
                        <th width="20%">Feature Subcategory</th>
                        <th width="40%">Feature Name</th>
                        <th width="20%">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Parallel Features - FSDP2 -->
                    <tr>
                        <td rowspan="10" class="category-cell">Parallel Features</td>
                        <td rowspan="1" class="subcategory-cell">FSDP2</td>
                        <td class="feature-name">FSDP2</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Parallel Features - TP Parallelism -->
                    <tr>
                        <td rowspan="1" class="subcategory-cell">TP Parallelism</td>
                        <td class="feature-name">tensor-parallel</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Parallel Features - PP Parallelism -->
                    <tr>
                        <td rowspan="3" class="subcategory-cell">PP Parallelism</td>
                        <td class="feature-name">PP</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <tr>
                        <td class="feature-name">Multi-parameter PP</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <tr>
                        <td class="feature-name">Dynamic PP</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Parallel Features - Sequence Parallelism -->
                    <tr>
                        <td rowspan="4" class="subcategory-cell">Sequence Parallelism</td>
                        <td class="feature-name">Ulysses</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <tr>
                        <td class="feature-name">RingAttention</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <tr>
                        <td class="feature-name">USP</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <tr>
                        <td class="feature-name">TP-SP</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Parallel Features - Heterogeneous Parallelism -->
                    <tr>
                        <td rowspan="1" class="subcategory-cell">Heterogeneous Parallelism</td>
                        <td class="feature-name">hetero-parallel</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Memory Optimization - Recomputation -->
                    <tr>
                        <td rowspan="2" class="category-cell">Memory Optimization</td>
                        <td class="subcategory-cell">Recomputation</td>
                        <td class="feature-name empty-cell">-</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Memory Optimization - Offload -->
                    <tr>
                        <td class="subcategory-cell">Offload</td>
                        <td class="feature-name">async offload</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                    <!-- Data Optimization - Load Balancing -->
                    <tr>
                        <td rowspan="1" class="category-cell">Memory Optimization</td>
                        <td class="subcategory-cell">Load Balancing</td>
                        <td class="feature-name">Online Data Reordering</td>
                        <td class="status-cell"><span class="status-released">Released</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
