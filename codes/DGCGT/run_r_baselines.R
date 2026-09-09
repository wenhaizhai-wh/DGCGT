args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: Rscript run_r_baselines.R expr.csv labels.csv out.csv [scores.csv] [raceid_mintotal] [raceid_clustnr] [raceid_bootnr]")

options(stringsAsFactors = FALSE)

has_numeric_column_header <- function(path) {
  first_line <- readLines(path, n = 1, warn = FALSE)
  tokens <- strsplit(first_line, ",", fixed = TRUE)[[1]]
  values <- suppressWarnings(as.numeric(tokens))
  length(values) > 1 && all(is.finite(values)) && all(values == seq_along(values) - 1)
}

has_header <- has_numeric_column_header(args[1])
expr_cells <- as.matrix(read.csv(args[1], header = has_header, check.names = FALSE))
storage.mode(expr_cells) <- "numeric"
labels <- read.csv(args[2], header = has_header, check.names = FALSE)[[1]]
if (nrow(expr_cells) != length(labels)) stop(sprintf("rows/labels mismatch: %d vs %d", nrow(expr_cells), length(labels)))
if (any(!is.finite(expr_cells))) stop("expression matrix contains invalid values")

out_path <- args[3]
score_path <- if (length(args) >= 4) args[4] else sub("\\.csv$", "_scores.csv", out_path)
raceid_mintotal <- if (length(args) >= 5) as.numeric(args[5]) else 3000
raceid_clustnr <- if (length(args) >= 6) as.numeric(args[6]) else 30
raceid_bootnr <- if (length(args) >= 7) as.numeric(args[7]) else 50

y <- as.integer(labels == 1)
expr <- t(expr_cells)
rownames(expr) <- paste0("gene", seq_len(nrow(expr)))
colnames(expr) <- paste0("cell", seq_len(ncol(expr)))

f1_from_pred <- function(y, pred) {
  tp <- sum(y == 1 & pred == 1)
  precision <- tp / max(sum(pred == 1), 1)
  recall <- tp / max(sum(y == 1), 1)
  if (precision + recall == 0) return(0)
  2 * precision * recall / (precision + recall)
}

average_precision <- function(y, score) {
  positive <- sum(y == 1)
  negative <- sum(y == 0)
  if (positive == 0 || negative == 0) return(0)
  order_index <- order(score, decreasing = TRUE, na.last = NA)
  yy <- y[order_index]
  ss <- score[order_index]
  if (length(yy) == 0) return(0)
  end_index <- c(which(diff(ss) != 0), length(ss))
  tp <- cumsum(yy == 1)[end_index]
  fp <- cumsum(yy == 0)[end_index]
  precision <- tp / pmax(tp + fp, 1)
  recall <- tp / positive
  previous_recall <- c(0, recall[-length(recall)])
  sum((recall - previous_recall) * precision)
}

roc_auc <- function(y, score) {
  positive <- sum(y == 1)
  negative <- sum(y == 0)
  if (positive == 0 || negative == 0) return(0)
  ranks <- rank(score, ties.method = "average")
  positive_rank_sum <- sum(ranks[y == 1])
  (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)
}

metric_row <- function(method, score, status = "success") {
  score <- as.numeric(score)
  if (length(score) != length(y)) stop(sprintf("%s score length mismatch: %d vs %d", method, length(score), length(y)))
  score[!is.finite(score)] <- 0
  pred <- as.integer(score >= 0.5)
  tp <- sum(y == 1 & pred == 1)
  data.frame(
    method = method,
    status = status,
    n_cells = length(y),
    positive_count = sum(y),
    thr_f1 = f1_from_pred(y, pred),
    precision = tp / max(sum(pred == 1), 1),
    recall = tp / max(sum(y == 1), 1),
    auprc = average_precision(y, score),
    roc_auc = roc_auc(y, score),
    stringsAsFactors = FALSE
  )
}

results <- list()
all_scores <- list()
errors <- character(0)

run_method <- function(method, expression) {
  score <- tryCatch(expression, error = function(e) {
    errors <<- c(errors, paste(method, conditionMessage(e), sep = ": "))
    NULL
  })
  if (!is.null(score)) {
    results[[length(results) + 1]] <<- metric_row(method, score)
    all_scores[[length(all_scores) + 1]] <<- data.frame(method = method, score = as.numeric(score))
  }
}

if (requireNamespace("FiRE", quietly = TRUE)) {
  run_method("FiRE", {
    model <- methods::new(FiRE::FiRE, 100, 50)
    model$fit(expr_cells)
    as.numeric(model$score(expr_cells))
  })
} else {
  errors <- c(errors, "FiRE: package is not installed")
}

if (requireNamespace("GapClust", quietly = TRUE)) {
  seurat_ns <- asNamespace("Seurat")
  unlockBinding("FindVariableFeatures.Assay", seurat_ns)
  patched_fvf <- function(object, selection.method = "vst", loess.span = 0.3,
                          clip.max = "auto", mean.function = Seurat::FastExpMean,
                          dispersion.function = Seurat::FastLogVMR, num.bin = 20,
                          binning.method = "equal_width", nfeatures = 2000,
                          mean.cutoff = c(0.1, 8), dispersion.cutoff = c(1, Inf),
                          verbose = TRUE, ...) {
    if (length(mean.cutoff) != 2 || length(dispersion.cutoff) != 2) {
      stop("Both mean.cutoff and dispersion.cutoff must be two numbers")
    }
    if (selection.method == "vst") {
      data <- SeuratObject::GetAssayData(object = object, layer = "counts")
      if (SeuratObject::IsMatrixEmpty(x = data)) data <- SeuratObject::GetAssayData(object = object, layer = "data")
    } else {
      data <- SeuratObject::GetAssayData(object = object, layer = "data")
    }
    hvf.info <- Seurat::FindVariableFeatures(
      object = data, selection.method = selection.method, loess.span = loess.span,
      clip.max = clip.max, mean.function = mean.function,
      dispersion.function = dispersion.function, num.bin = num.bin,
      binning.method = binning.method, nfeatures = nfeatures,
      mean.cutoff = mean.cutoff, dispersion.cutoff = dispersion.cutoff,
      verbose = verbose, ...
    )
    object[[names(x = hvf.info)]] <- hvf.info
    hvf.info <- hvf.info[which(hvf.info[, 1, drop = TRUE] != 0), , drop = FALSE]
    if (selection.method == "vst") {
      hvf.info <- hvf.info[order(hvf.info$vst.variance.standardized, decreasing = TRUE), , drop = FALSE]
    } else {
      hvf.info <- hvf.info[order(hvf.info$mvp.dispersion, decreasing = TRUE), , drop = FALSE]
    }
    selection.method <- switch(EXPR = selection.method, mvp = "mean.var.plot", disp = "dispersion", selection.method)
    top.features <- switch(
      EXPR = selection.method,
      mean.var.plot = {
        means.use <- (hvf.info[, 1] > mean.cutoff[1]) & (hvf.info[, 1] < mean.cutoff[2])
        dispersions.use <- (hvf.info[, 3] > dispersion.cutoff[1]) & (hvf.info[, 3] < dispersion.cutoff[2])
        rownames(hvf.info)[which(means.use & dispersions.use)]
      },
      dispersion = head(rownames(hvf.info), nfeatures),
      vst = head(rownames(hvf.info), nfeatures),
      stop("Unknown selection method: ", selection.method)
    )
    Seurat::VariableFeatures(object) <- top.features
    vf.name <- paste0(ifelse(selection.method == "vst", "vst", "mvp"), ".variable")
    object[[vf.name]] <- rownames(object[[]]) %in% top.features
    object
  }
  assign("FindVariableFeatures.Assay", patched_fvf, envir = seurat_ns)
  lockBinding("FindVariableFeatures.Assay", seurat_ns)

  gap_imports <- parent.env(asNamespace("GapClust"))
  if (exists("Neighbour", envir = gap_imports, inherits = FALSE)) unlockBinding("Neighbour", gap_imports)
  assign("Neighbour", function(query, ref, k, build = "kdtree", cores = 0, checks = 1) {
    rflann::Neighbour(query = query, ref = ref, k = k, build = build, cores = cores, checks = checks)
  }, envir = gap_imports)
  lockBinding("Neighbour", gap_imports)

  run_method("GapClust", {
    res <- GapClust::GapClust(expr, k = 200)
    score <- rep(0, ncol(expr))
    if (!is.null(res$rare_score) && !is.null(res$skewness)) {
      rare_columns <- which(res$skewness > 2)
      if (length(rare_columns) == 0) rare_columns <- seq_len(ncol(res$rare_score))
      score <- apply(res$rare_score[, rare_columns, drop = FALSE], 1, max)
    } else if (!is.null(res$rare_index)) {
      score[res$rare_index] <- 1
    }
    score
  })
} else {
  errors <- c(errors, "GapClust: package is not installed")
}

if (requireNamespace("RaceID", quietly = TRUE)) {
  run_method("RaceID", {
    sc <- RaceID::SCseq(expr)
    sc <- RaceID::filterdata(sc, mintotal = raceid_mintotal, verbose = TRUE)
    cat("RaceID retained cells:", ncol(sc@ndata), "\n")
    sc <- RaceID::compdist(sc, metric = "pearson", FSelect = TRUE, no_cores = 1)
    sc <- RaceID::clustexp(sc, clustnr = raceid_clustnr, bootnr = raceid_bootnr, rseed = 17000, verbose = TRUE)
    sc <- RaceID::findoutliers(sc, verbose = TRUE)
    score <- integer(length(y))
    out_names <- sc@out$out
    if (!is.null(out_names)) score[match(out_names, colnames(sc@expdata), nomatch = 0)] <- 1
    score
  })
} else {
  errors <- c(errors, "RaceID: package is not installed")
}

result_frame <- if (length(results) > 0) do.call(rbind, results) else data.frame()
write.csv(result_frame, out_path, row.names = FALSE)
if (length(all_scores) > 0) write.csv(do.call(rbind, all_scores), score_path, row.names = FALSE)
error_path <- sub("\\.csv$", "_errors.txt", out_path)
writeLines(errors, error_path)
print(result_frame)
if (length(errors) > 0) cat(paste(errors, collapse = "\n"), "\n")


