(function () {
    const state = {
        documentId: null,
        previewUrl: "#",
        answerText: "",
        answerSnippet: "",
        selectedSectionId: "introduction",
        sections: [],
    };

    const elements = {
        uploadButton: document.getElementById("upload-button"),
        fileInput: document.getElementById("file-input"),
        questionForm: document.getElementById("question-form"),
        questionInput: document.getElementById("question-input"),
        askButton: document.getElementById("ask-button"),
        statusChip: document.getElementById("status-chip"),
        statusLabel: document.getElementById("status-label"),
        documentName: document.getElementById("document-name"),
        documentPages: document.getElementById("document-pages"),
        documentType: document.getElementById("document-type"),
        previewLink: document.getElementById("preview-link"),
        answerQuote: document.getElementById("answer-quote"),
        answerSnippet: document.getElementById("answer-snippet"),
        supportingPoints: document.getElementById("supporting-points"),
        analysisSubtitle: document.getElementById("analysis-subtitle"),
        verificationLabel: document.getElementById("verification-label"),
        relatedContext: document.getElementById("related-context"),
        refineCopy: document.getElementById("refine-copy"),
        annotationsList: document.getElementById("annotations-list"),
        feedbackBanner: document.getElementById("feedback-banner"),
        copyAnswer: document.getElementById("copy-answer"),
        dropZone: document.getElementById("supporting-drop"),
        chapterItems: Array.from(document.querySelectorAll(".chapter-item")),
    };

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function setBusy(isBusy, label) {
        elements.statusChip.classList.toggle("busy", isBusy);
        elements.statusChip.classList.toggle("ready", !isBusy);
        elements.statusLabel.textContent = label;
        elements.askButton.disabled = isBusy;
        elements.uploadButton.disabled = isBusy;
    }

    function showBanner(message, isError) {
        elements.feedbackBanner.textContent = message;
        elements.feedbackBanner.classList.add("show");
        elements.feedbackBanner.classList.toggle("bg-primary", !isError);
        elements.feedbackBanner.classList.toggle("text-on-primary", !isError);
        elements.feedbackBanner.classList.toggle("bg-error", isError);
        elements.feedbackBanner.classList.toggle("text-on-error", isError);
        window.clearTimeout(showBanner.timer);
        showBanner.timer = window.setTimeout(() => {
            elements.feedbackBanner.classList.remove("show");
        }, 2600);
    }

    function buildSupportingList(items) {
        if (!items.length) {
            return `
                <li class="flex gap-4 items-start result-item">
                    <span class="w-6 h-6 rounded-full bg-surface-container-high flex-shrink-0 flex items-center justify-center text-[10px] font-bold">01</span>
                    <span class="text-on-surface-variant">No supporting excerpt was extracted from this document.</span>
                </li>
            `;
        }

        return items
            .slice(0, 2)
            .map((item, index) => {
                const number = String(index + 1).padStart(2, "0");
                return `
                    <li class="flex gap-4 items-start result-item">
                        <span class="w-6 h-6 rounded-full bg-surface-container-high flex-shrink-0 flex items-center justify-center text-[10px] font-bold">${number}</span>
                        <span class="text-on-surface-variant">${escapeHtml(item)}</span>
                    </li>
                `;
            })
            .join("");
    }

    function sectionById(sectionId) {
        return state.sections.find((section) => section.section_id === sectionId) || null;
    }

    function setActiveChapter(sectionId, announce) {
        state.selectedSectionId = sectionId;
        elements.chapterItems.forEach((item) => {
            const isActive = item.dataset.sectionId === sectionId;
            item.classList.toggle("active", isActive);
            item.classList.toggle("inactive", !isActive);
        });

        const activeSection = sectionById(sectionId);
        if (activeSection) {
            elements.analysisSubtitle.textContent = `${activeSection.title} focus`;
            elements.relatedContext.textContent = activeSection.excerpt || "No excerpt available for this section yet.";
            elements.refineCopy.textContent = activeSection.keywords && activeSection.keywords.length
                ? `Transformer focus: ${activeSection.keywords.join(", ")}.`
                : `Transformer focus on ${activeSection.title.toLowerCase()}.`;
            renderAnnotations([activeSection.excerpt].filter(Boolean));
            if (announce) {
                showBanner(`Selected ${activeSection.title}.`, false);
            }
        }
    }

    function renderSections(sections) {
        state.sections = sections || [];
        elements.chapterItems.forEach((item) => {
            const section = state.sections.find((entry) => entry.section_id === item.dataset.sectionId);
            const titleNode = item.querySelector(".chapter-title");
            const iconNode = item.querySelector(".chapter-icon");
            titleNode.textContent = section ? section.title : item.dataset.defaultTitle;
            if (section && section.icon) {
                iconNode.textContent = section.icon;
            }
        });

        const preferredSection = sectionById(state.selectedSectionId) ? state.selectedSectionId : "introduction";
        setActiveChapter(preferredSection, false);
    }

    function renderAnnotations(items) {
        const notes = items.slice(0, 2);
        if (!notes.length) {
            elements.annotationsList.innerHTML = `
                <div class="bg-surface-container-lowest p-4 rounded shadow-[0_2px_8px_rgba(0,0,0,0.02)] space-y-2 annotation-card">
                    <div class="flex justify-between items-start">
                        <span class="text-[10px] font-bold text-tertiary px-2 py-0.5 bg-tertiary-fixed rounded">P. 01</span>
                        <span class="text-[10px] text-secondary">now</span>
                    </div>
                    <p class="text-xs text-on-surface-variant leading-relaxed annotation-text">Upload a document to generate annotations.</p>
                </div>
            `;
            return;
        }

        elements.annotationsList.innerHTML = notes
            .map((item, index) => {
                const badgeClass = index === 0 ? "bg-tertiary-fixed" : "bg-surface-container-high";
                const wrapperClass = index === 1 ? " opacity-50" : "";
                return `
                    <div class="bg-surface-container-lowest p-4 rounded shadow-[0_2px_8px_rgba(0,0,0,0.02)] space-y-2 annotation-card${wrapperClass}">
                        <div class="flex justify-between items-start">
                            <span class="text-[10px] font-bold text-tertiary px-2 py-0.5 ${badgeClass} rounded">P. ${String(index + 1).padStart(2, "0")}</span>
                            <span class="text-[10px] text-secondary">${index === 0 ? "now" : "earlier"}</span>
                        </div>
                        <p class="text-xs text-on-surface-variant leading-relaxed annotation-text">${escapeHtml(item)}</p>
                    </div>
                `;
            })
            .join("");
    }

    function updateDocument(payload) {
        state.documentId = payload.document_id;
        state.previewUrl = payload.preview_url;
        elements.documentName.textContent = payload.name;
        elements.documentPages.textContent = `${payload.page_count} Pages`;
        elements.documentType.textContent = `${payload.type} Document`;
        elements.previewLink.href = payload.preview_url;

        renderSections(payload.sections || []);

        const keywords = payload.keywords || [];
        const snapshots = payload.snapshot_sentences || [];
        if (!state.sections.length) {
            elements.analysisSubtitle.textContent = keywords.length
                ? `Focus on ${keywords.slice(0, 2).join(" and ")}`
                : "Key insights from uploaded document";
            elements.relatedContext.textContent = snapshots[0] || "Ask a question to generate related context.";
            elements.refineCopy.textContent = keywords.length
                ? `Automate a summary for ${keywords.slice(0, 3).join(", ")}.`
                : "Automate a summary of the most relevant citations in this document.";
            renderAnnotations(snapshots);
        }
    }

    function updateAnswer(resultPayload, snapshots, keywords) {
        const result = resultPayload.result;
        const quote = result.answer || "No confident transformer answer found for that question.";
        const snippet = result.snippet || quote;

        state.answerText = quote;
        state.answerSnippet = snippet;

        elements.answerQuote.textContent = `"${quote}"`;
        elements.answerSnippet.textContent = snippet;
        elements.supportingPoints.innerHTML = buildSupportingList(snapshots);
        elements.verificationLabel.textContent = result.confidence
            ? `${result.confidence} confidence transformer answer`
            : "Verified against uploaded document";
        if (resultPayload.active_section) {
            elements.analysisSubtitle.textContent = `${resultPayload.active_section.title} analysis`;
        } else if (keywords && keywords.length) {
            elements.analysisSubtitle.textContent = `Focus on ${keywords.slice(0, 2).join(" and ")}`;
        }
        elements.relatedContext.textContent = snapshots[0] || elements.relatedContext.textContent;
        renderAnnotations(snapshots);
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);

        setBusy(true, "Uploading");
        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "Upload failed.");
            }

            updateDocument(payload);
            showBanner("Document uploaded and processed.", false);
        } catch (error) {
            showBanner(error.message, true);
        } finally {
            setBusy(false, "Ready");
            elements.fileInput.value = "";
        }
    }

    async function askQuestion(question) {
        if (!state.documentId) {
            showBanner("Upload a PDF or TXT document first.", true);
            return;
        }

        setBusy(true, "Thinking");
        try {
            const response = await fetch("/api/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    document_id: state.documentId,
                    question,
                    section_id: state.selectedSectionId,
                }),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "Question failed.");
            }

            updateAnswer(payload, payload.snapshot_sentences || [], payload.keywords || []);
            showBanner("Answer generated.", false);
        } catch (error) {
            showBanner(error.message, true);
        } finally {
            setBusy(false, "Ready");
        }
    }

    elements.uploadButton.addEventListener("click", () => {
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener("change", (event) => {
        const file = event.target.files && event.target.files[0];
        if (file) {
            uploadFile(file);
        }
    });

    elements.questionForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const question = elements.questionInput.value.trim();
        if (!question) {
            showBanner("Enter a question first.", true);
            return;
        }
        askQuestion(question);
    });

    elements.chapterItems.forEach((item) => {
        item.addEventListener("click", () => {
            if (!state.documentId) {
                showBanner("Upload a document first.", true);
                return;
            }
            setActiveChapter(item.dataset.sectionId, true);
        });
    });

    elements.copyAnswer.addEventListener("click", async () => {
        const text = `${state.answerText}\n\n${state.answerSnippet}`.trim();
        if (!text) {
            showBanner("No answer available to copy.", true);
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            showBanner("Answer copied to clipboard.", false);
        } catch (_error) {
            showBanner("Clipboard copy failed.", true);
        }
    });

    elements.dropZone.addEventListener("click", () => {
        elements.fileInput.click();
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        elements.dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            elements.dropZone.classList.add("bg-surface-container");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        elements.dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            elements.dropZone.classList.remove("bg-surface-container");
        });
    });

    elements.dropZone.addEventListener("drop", (event) => {
        const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) {
            uploadFile(file);
        }
    });
})();
