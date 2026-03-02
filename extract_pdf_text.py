import fitz

# Open PDF
pdf_path = r'd:\Uni_Documents\nam4_HKI\KLTN\LPNC_2\SCGI.pdf'
doc = fitz.open(pdf_path)

# Extract text from all pages
full_text = []
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    full_text.append(f"\n{'='*50}\nPage {page_num + 1}\n{'='*50}\n{text}")

# Save to file
output_path = r'd:\Uni_Documents\nam4_HKI\KLTN\LPNC_2\SCGI_extracted.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(full_text))

print(f"Extracted {len(doc)} pages to {output_path}")
print(f"Total characters: {sum(len(t) for t in full_text)}")

# Print first 5000 characters for preview
print("\n" + "="*50)
print("PREVIEW (first 5000 chars):")
print("="*50)
print('\n'.join(full_text)[:5000])
