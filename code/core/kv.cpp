#include "lm/lm.h"

#include <cstdlib>
#include <cstring>

struct lm_kv_storage {
    uint32_t tokens;
    uint32_t references;
    uint8_t used;
};

struct lm_kv_page {
    uint32_t storage_id;
    uint8_t used;
};

struct lm_kv_cache {
    uint32_t page_count;
    uint32_t page_tokens;
    uint32_t free_pages;
    uint32_t free_storages;
    uint64_t appended_tokens;
    lm_kv_page *pages;
    lm_kv_storage *storages;
};

static int valid_page(const lm_kv_cache *cache, uint32_t page_id) {
    return cache && page_id < cache->page_count && cache->pages[page_id].used &&
           cache->pages[page_id].storage_id < cache->page_count &&
           cache->storages[cache->pages[page_id].storage_id].used;
}

static uint32_t find_free(const lm_kv_storage *items, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) if (!items[i].used) return i;
    return UINT32_MAX;
}

static lm_status make_storage(lm_kv_cache *cache, uint32_t tokens, uint32_t *out_id) {
    if (!cache || !out_id || cache->free_storages == 0u) return LM_ERR_CAPACITY;
    const uint32_t id = find_free(cache->storages, cache->page_count);
    if (id == UINT32_MAX) return LM_ERR_CAPACITY;
    cache->storages[id].used = 1u;
    cache->storages[id].tokens = tokens;
    cache->storages[id].references = 1u;
    cache->free_storages--;
    *out_id = id;
    return LM_OK;
}

static lm_status make_page_safe(lm_kv_cache *cache, uint32_t storage_id, uint32_t *out_id) {
    if (!cache || !out_id || cache->free_pages == 0u) return LM_ERR_CAPACITY;
    uint32_t id = UINT32_MAX;
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        if (!cache->pages[i].used) { id = i; break; }
    }
    if (id == UINT32_MAX) return LM_ERR_CAPACITY;
    cache->pages[id].used = 1u;
    cache->pages[id].storage_id = storage_id;
    cache->free_pages--;
    *out_id = id;
    return LM_OK;
}

static lm_status detach_for_write(lm_kv_cache *cache, uint32_t page_id) {
    if (!valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    lm_kv_page *page = &cache->pages[page_id];
    lm_kv_storage *old = &cache->storages[page->storage_id];
    if (old->references <= 1u) return LM_OK;
    uint32_t replacement = UINT32_MAX;
    const lm_status status = make_storage(cache, old->tokens, &replacement);
    if (status != LM_OK) return status;
    old->references--;
    page->storage_id = replacement;
    return LM_OK;
}

lm_status lm_kv_cache_create(uint32_t page_count, uint32_t page_tokens,
                             lm_kv_cache **out_cache) {
    if (!out_cache || page_count == 0u || page_tokens == 0u) return LM_ERR_ARGUMENT;
    *out_cache = nullptr;
    lm_kv_cache *cache = static_cast<lm_kv_cache *>(std::calloc(1u, sizeof(*cache)));
    if (!cache) return LM_ERR_CAPACITY;
    cache->pages = static_cast<lm_kv_page *>(std::calloc(page_count, sizeof(*cache->pages)));
    cache->storages = static_cast<lm_kv_storage *>(std::calloc(page_count, sizeof(*cache->storages)));
    if (!cache->pages || !cache->storages) {
        std::free(cache->pages);
        std::free(cache->storages);
        std::free(cache);
        return LM_ERR_CAPACITY;
    }
    cache->page_count = page_count;
    cache->page_tokens = page_tokens;
    cache->free_pages = page_count;
    cache->free_storages = page_count;
    *out_cache = cache;
    return LM_OK;
}

void lm_kv_cache_destroy(lm_kv_cache *cache) {
    if (!cache) return;
    std::free(cache->pages);
    std::free(cache->storages);
    std::free(cache);
}

lm_status lm_kv_cache_append(lm_kv_cache *cache, uint32_t *page_id,
                             uint32_t token_count) {
    if (!cache || !page_id || token_count == 0u || token_count > cache->page_tokens)
        return LM_ERR_ARGUMENT;
    if (*page_id == UINT32_MAX) {
        uint32_t storage_id = UINT32_MAX;
        lm_status status = make_storage(cache, 0u, &storage_id);
        if (status != LM_OK) return status;
        status = make_page_safe(cache, storage_id, page_id);
        if (status != LM_OK) {
            std::memset(&cache->storages[storage_id], 0, sizeof(cache->storages[storage_id]));
            cache->free_storages++;
            return status;
        }
    }
    if (!valid_page(cache, *page_id)) return LM_ERR_ARGUMENT;
    lm_status status = detach_for_write(cache, *page_id);
    if (status != LM_OK) return status;
    lm_kv_storage *storage = &cache->storages[cache->pages[*page_id].storage_id];
    if (storage->tokens + token_count > cache->page_tokens) return LM_ERR_CAPACITY;
    storage->tokens += token_count;
    cache->appended_tokens += token_count;
    return LM_OK;
}

lm_status lm_kv_cache_fork(lm_kv_cache *cache, uint32_t source_page,
                           uint32_t *out_page) {
    if (!valid_page(cache, source_page) || !out_page) return LM_ERR_ARGUMENT;
    lm_kv_storage *storage = &cache->storages[cache->pages[source_page].storage_id];
    lm_status status = make_page_safe(cache, cache->pages[source_page].storage_id, out_page);
    if (status != LM_OK) return status;
    storage->references++;
    return LM_OK;
}

lm_status lm_kv_cache_rollback(lm_kv_cache *cache, uint32_t page_id,
                               uint32_t token_count) {
    if (!valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    lm_status status = detach_for_write(cache, page_id);
    if (status != LM_OK) return status;
    lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    if (token_count > storage->tokens) return LM_ERR_ARGUMENT;
    storage->tokens -= token_count;
    if (cache->appended_tokens >= token_count) cache->appended_tokens -= token_count;
    return LM_OK;
}

lm_status lm_kv_cache_release(lm_kv_cache *cache, uint32_t page_id) {
    if (!valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    lm_kv_page *page = &cache->pages[page_id];
    lm_kv_storage *storage = &cache->storages[page->storage_id];
    if (storage->references == 0u) return LM_ERR_STATE;
    storage->references--;
    if (storage->references == 0u) {
        std::memset(storage, 0, sizeof(*storage));
        cache->free_storages++;
    }
    std::memset(page, 0, sizeof(*page));
    cache->free_pages++;
    return LM_OK;
}

lm_status lm_kv_cache_get_stats(const lm_kv_cache *cache, lm_kv_stats *out_stats) {
    if (!cache || !out_stats) return LM_ERR_ARGUMENT;
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->page_tokens = cache->page_tokens;
    out_stats->total_pages = cache->page_count;
    out_stats->free_pages = cache->free_pages;
    out_stats->appended_tokens = cache->appended_tokens;
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        if (cache->pages[i].used) out_stats->used_pages++;
    }
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        if (cache->storages[i].used && cache->storages[i].references > 1u)
            out_stats->shared_pages++;
    }
    return LM_OK;
}
