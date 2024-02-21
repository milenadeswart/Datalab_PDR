library(tidyverse)
library(readxl)
library(here)

zoekwoorden <- readxl::read_xlsx(here("datasets/Zoekwoorden 26.01.24.xlsx"),
                          sheet = 1, skip = 3) %>%
  janitor::clean_names() %>%
  rename(categorie = x1,
         toelichting = x6) %>%
  fill(categorie) %>%
  separate_rows(zoekwoord, sep = ",") %>%
  separate_rows(voorkeur, sep = ",") %>%
  separate_rows(vermijden, sep = ",") %>%
  mutate(across(where(is.character), str_trim)) %>%
  mutate(across(where(is.character), tolower)) %>%
  distinct()
