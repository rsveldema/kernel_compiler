#pragma once

#include <vulkan_session.hpp>

class VulkanTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_session = std::make_unique<VulkanSession>();
    }
    void TearDown() override { m_session = nullptr; }

    VkDevice get_device() const { return m_session->get_device(); }
    bool has_device() const { return m_session->has_device(); }

    VulkanSession& get_session() {
        assert(m_session);
        return *m_session;
    }

private:
    std::unique_ptr<VulkanSession> m_session;
};

