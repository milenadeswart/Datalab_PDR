library(tidyverse)
library(here)
library(pdftools)
library(tidytext)

# Stap 1
map_bestanden <- here("datasets/ruwe_data/")
lijst_bestanden <- tibble(bestandsmap = map_bestanden,
                          bestandsnaam = list.files(map_bestanden, recursive = T))

# Stap 2
inlezen_pdf_bestand <- function(bestandslocatie){
  pdftools::pdf_text(bestandslocatie)
}

inlezen_pdf_bestand_safely <- purrr::safely(inlezen_pdf_bestand)

# Stap 3
tekst_bestanden <- lijst_bestanden %>%
  filter(str_detect(bestandsnaam, ".pdf")) %>%
  # head(10)
  mutate(text = map(file.path(bestandsmap, bestandsnaam),
                    inlezen_pdf_bestand_safely))

# Stap 4
documenten <- tekst_bestanden %>%
  mutate(text = map(text, \(x) list(result = x$result, error = list(x$error)))) %>%
  unnest_wider(text) %>%
  unnest_longer(result) %>%
  rename(text = result) %>%
  select(-error) %>%
  unnest_longer(text) %>%
  group_by(bestandsnaam) %>%
  mutate(pagina = row_number()) %>%
  group_by(across(c(-text))) %>%
  unnest_tokens(input = text, output = text, token = "paragraphs", to_lower = F) %>%
  mutate(text = str_trim(text)) %>%
  mutate(paragraaf = row_number()) %>%
  select(bestandsmap, bestandsnaam, pagina, paragraaf, text)

# Opslaan van het bestand

saveRDS(documenten, "datasets/documenten.RDS")
