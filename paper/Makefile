MAIN = main
TEX  = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex refs.bib iclr2026_conference.sty iclr2026_conference.bst
	$(TEX) $(MAIN)
	bibtex $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).pdf *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz sections/*.aux
