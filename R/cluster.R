library(glue)
library(Rtsne)
library(RColorBrewer)

df <- read.csv("../python/curve_feats_counties.csv")
df <- df[, -which(grepl('2023', names(df)))]
ts <- read.csv("../python/with_geo_household_cnt.csv")

scaled_df <- apply(df[, -1], 2, scale, center=TRUE, scale=TRUE)
scaled_df <- as.data.frame(scaled_df)
scaled_df[['NAME']] <- df$X
head(scaled_df)

scaled_df[['no_curve_married']] <- (is.na(scaled_df[['married.slope_2022']]) - 0.5) * 2
scaled_df[['no_curve_unmarried']] <- (is.na(scaled_df[['unmarried.slope_2022']]) - 0.5) * 2
for(i in seq_len(ncol(scaled_df))){
  
  #is_na_vals <- is.na(scaled_df[, i])
  #scaled_df[is_na_vals, i] <- 0 # replaced with mean, just so kmeans will run!
  
  #Technical implementation 1: Imputation with column means
  na_idx <- is.na(scaled_df[[i]])
  if (any(na_idx)) {
    col_mean <- mean(scaled_df[[i]], na.rm = TRUE)
    scaled_df[[i]][na_idx] <- col_mean
  }
}


no_m_yes_u <- (scaled_df[['no_curve_married']] == 1) & (scaled_df[['no_curve_unmarried']] == -1)
yes_m_no_u <- (scaled_df[['no_curve_married']] == -1) & (scaled_df[['no_curve_unmarried']] == 1)
no_m_no_u <- (scaled_df[['no_curve_married']] == 1) & (scaled_df[['no_curve_unmarried']] == 1)
scenarios <- list(no_m_yes_u, yes_m_no_u, no_m_no_u)
png('look_at_bad_fits.png', 1200, 500)
par(mfrow=c(1, 3))
for(i in seq_along(scenarios)){
  s <- scenarios[[i]]
  rand_county <- sample(scaled_df[s, "NAME"], 1)
  sdf <- ts[ts$NAME == rand_county, c('year', 'B11002_003E', 'B11002_012E')]
  y_range <- c(min(sdf[, -1]), max(sdf[, -1]))
  plot(sdf$year, sdf[, 2], col="blue", pch=16, ylim=y_range)
  points(sdf$year, sdf[, 3], col="red", pch=16)
  legend("topleft", legend=c('married', 'unmarried'),
         fill=c('blue', 'red'))
}
dev.off()


# Technical Implementation 2: PCA
library(dplyr)
library(cluster)

# 1) Drop NAME column for PCA
feat_mat <- scaled_df[, !(names(scaled_df) %in% "NAME")]

# 2) PCA (keep first 5 components)
pca_res <- prcomp(feat_mat, center = FALSE, scale. = FALSE)
pc_df   <- as.data.frame(pca_res$x[, 1:5])

# 3) Compute average silhouette width for k = 2…10
sil_scores <- sapply(2:10, function(k) {
  km <- kmeans(pc_df, centers = k, nstart = 25)
  mean(silhouette(km$cluster, dist(pc_df))[, "sil_width"])
})

# 4) Choose best k
opt_k <- which.max(sil_scores) + 1
message("Auto-selected number of clusters: ", opt_k)

# 5) Final k-means on PC space
km_out <- kmeans(pc_df, centers = opt_k, nstart = 25)



scaled_df <- scaled_df[scaled_df$NAME != 'Los Angeles County, California', ]
ks <- 2:20
km_bag <- matrix(NA, ncol=3, nrow=length(ks))
for(k in ks){
  km_out <- kmeans(scaled_df[, -which(names(scaled_df) == 'NAME')], centers=k, nstart=20)
  km_bag[k - 1, ] <- c(k, km_out$tot.withinss, km_out$betweenss)
}

png('kmeans_btwss_by_k3.png')
plot(km_bag[, 1], km_bag[, 3])
dev.off()

km_bag
km_out <- kmeans(scaled_df[, -which(names(scaled_df) == 'NAME')], centers=4, nstart=20)
table(km_out$cluster) 
scaled_df[km_out$cluster == 3,] # Los Angeles is its own cluster!

cols4 <- brewer.pal(4, "Set1")
png('kmeans_4_centers_slope.png')
par(mfrow=c(2, 2))
plot(km_out$centers[, 'married.val_2022'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'married.slope_2022'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'unmarried.val_2022'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'unmarried.slope_2022'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'married.slope_2022'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'unmarried.slope_2022'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'no_curve_married'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'no_curve_unmarried'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
dev.off()


cols <- brewer.pal(4, 'Set1')
for(p in seq(5, 35, by=10)){
  tsne_results <- Rtsne(scaled_df,
                        dims = 2, perplexity = p, verbose = TRUE, max_iter = 2500)
  png(glue('perplex_{p}_tsne_iter2500.png'))
  plot(tsne_results$Y[, 1], tsne_results$Y[, 2], main=glue("Perplexity {p}"),
       col=cols[km_out$cluster])
  dev.off()
}

scaled_df[['cluster']] <- km_out$cluster
write.csv(scaled_df, 'cluster_out.csv')


# before: target_vars on slopes
# target_vars <- list(
#   c('married.slope_2022',   'B11002_003E'),
#   c('unmarried.slope_2022', 'B11002_012E')
# )

# updated to use the no_curve flags
target_vars <- list(
  c('no_curve_married',   'B11002_003E'),
  c('no_curve_unmarried', 'B11002_012E')
)

for (vs in target_vars) {
  flag_var   <- vs[1]        # 'no_curve_married' or 'no_curve_unmarried'
  target_var <- vs[2]        # the raw count column in ts
  grp        <- sub('no_curve_', '', flag_var)
  
  # sample one "bad‐fit" county where flag == +1
  bad_idx    <- scaled_df[[flag_var]] ==  1
  county_vec <- scaled_df$NAME[bad_idx]
  
  if (length(county_vec) == 0) {
    message("No counties flagged as bad for ", grp)
    next
  }
  
  rand_county <- sample(county_vec, 1)
  sdf         <- ts[ts$NAME == rand_county, c('year', target_var)]
  
  png(glue("eg_{flag_var}_curve.png"))
  plot(
    sdf$year, sdf[[target_var]],
    main = glue("No best‐fit curve for {grp} households\n{rand_county}")
  )
  dev.off()
}


# Vis the clusters
for(clus in 1:4){
  # get 3 counties
  rand_counties <- sample(scaled_df[km_out$cluster == clus, 'NAME'], 5)
  for(i in seq_along(rand_counties)){
    rc = rand_counties[i]
    sdf <- ts[ts$NAME == rc, c('year', 'B11002_003E', 'B11002_012E')]
    y1 <- sdf[['B11002_003E']]
    y1 <- (y1 - min(y1)) / (max(y1) - min(y1))
    y2 <- sdf[['B11002_012E']]
    y2 <- (y2 - min(y2)) / (max(y2) - min(y2))
    png(glue('eg_cluster{clus}_{i}_curve.png'))
    plot(sdf$year, y1, main=glue("Cluster {clus} household count\n {rc}"), col="blue", pch=16, ylim=c(0, 1))
    points(sdf$year, y2, col="red", pch=16)
    legend("topleft", fill=c("blue", "red"), legend=c("married", "unmarried"), pch=16, title='household count')
    dev.off()
  }
}
# large variation in values


