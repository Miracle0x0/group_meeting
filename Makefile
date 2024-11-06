CUSTOM_OPTIONS := --theme white --css ./css/style.css --separator "\n===\n" --vertical-separator "\n====\n"
PRINT_OPTIONS := --theme white --css ./css/style_for_print.css --separator "\n===\n" --vertical-separator "\n====\n"

.PHONY: run share pdf clean
run:
	reveal-md main.md -w $(CUSTOM_OPTIONS)

share:
	reveal-md main.md -w $(CUSTOM_OPTIONS) --host 0.0.0.0 --disable-auto-open

pdf: main.md
	reveal-md main.md $(PRINT_OPTIONS) --disable-auto-open --port 11451 --print meeting_slides.pdf

clean:
	rm -f *.pdf
